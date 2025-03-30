import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def load_model(model_path):
    """
    Load a trained model from disk

    Parameters:
    model_path (str): Path to the saved model file

    Returns:
    object: Loaded model
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model


def prepare_match_data(match_data):
    """
    Prepare a single match's data for prediction

    Parameters:
    match_data (dict): Dictionary containing match statistics

    Returns:
    pandas.DataFrame: DataFrame ready for prediction
    """
    # Convert match data to DataFrame
    df = pd.DataFrame([match_data])

    # Ensure map is encoded the same way as in training
    map_mapping = {
        'Mirage': 0,
        'Dust2': 1,
        'Inferno': 2,
        'Nuke': 3,
        'Overpass': 4,
        'Train': 5,
        'Vertigo': 6,
        'Cache': 7,
        'Cobblestone': 8,
        'Ancient': 9
    }

    if 'map' in df.columns and df['map'].dtype == 'object':
        df['map'] = df['map'].map(map_mapping)

    # Feature engineering (same as in training)
    # Calculate K/D ratio
    df['kd_ratio'] = df['kills'] / df['deaths'].replace(0, 1)  # Avoid division by zero

    # Calculate headshot efficiency
    df['hs_efficiency'] = df['hs_percent'] * df['kills'] / 100

    # Calculate impact score
    df['impact_score'] = (df['kills'] + df['assists'] * 0.5 + df['mvps'] * 2) / (df['deaths'] + 1)

    # Calculate KAST approximation
    df['kast_approx'] = (df['kills'] + df['assists']) / (df['deaths'] + df['kills'] + df['assists'])

    print("Match data prepared successfully!")
    return df


def predict_match_result(model, match_df):
    """
    Predict the result of a match

    Parameters:
    model: Trained model
    match_df (pandas.DataFrame): DataFrame with match statistics

    Returns:
    tuple: Predicted result and probability
    """
    # Make prediction
    if 'result' in match_df.columns:
        match_df = match_df.drop('result', axis=1)

    prediction = model.predict(match_df)
    probabilities = model.predict_proba(match_df)

    result_mapping = {0: "Loss", 1: "Win"}
    predicted_result = result_mapping.get(prediction[0], f"Class {prediction[0]}")

    print(f"Prediction complete!")
    return predicted_result, probabilities


def explain_prediction(match_data, predicted_result, probabilities):
    """
    Explain the prediction in human-readable terms

    Parameters:
    match_data (dict): Original match data
    predicted_result (str): Predicted match result
    probabilities (array): Prediction probabilities

    Returns:
    str: Explanation of prediction
    """
    # Generate explanation based on key stats and model probabilities
    max_prob = max(probabilities[0]) * 100

    explanation = f"Prediction: {predicted_result} (Confidence: {max_prob:.1f}%)\n\n"
    explanation += "Key performance factors:\n"

    # Add key statistics that influenced the prediction
    kd_ratio = match_data['kills'] / max(1, match_data['deaths'])
    explanation += f"- K/D Ratio: {kd_ratio:.2f}\n"
    explanation += f"- Kills: {match_data['kills']}\n"
    explanation += f"- Deaths: {match_data['deaths']}\n"
    explanation += f"- Assists: {match_data['assists']}\n"
    explanation += f"- MVPs: {match_data['mvps']}\n"
    explanation += f"- Headshot %: {match_data['hs_percent']}%\n"
    explanation += f"- Points: {match_data['points']}\n"
    explanation += f"- Map: {match_data['map']}"

    # Add interpretation
    explanation += "\n\nInterpretation:\n"
    if predicted_result == "Win":
        if max_prob > 75:
            explanation += "Your performance stats strongly indicate a victory."
        else:
            explanation += "Your performance suggests a win, but it could be a close match."
    else:
        if max_prob > 75:
            explanation += "Your performance stats strongly indicate a loss."
        else:
            explanation += "Your performance suggests a loss, but you might have a chance to turn it around."

    return explanation


def main():
    """
    Main function to demonstrate the prediction pipeline
    """
    print("\n===== CS:GO Match Result Predictor =====\n")

    # Try to load the model - if not found, provide instructions
    try:
        model = load_model("best_model_random_forest.joblib")  # Adjust filename as needed
    except FileNotFoundError:
        print("Model file not found! You need to run the training script first.")
        print("Example model filenames could be: best_model_random_forest.joblib, best_model_xgboost.joblib, etc.")
        return

    # Example match data (modify this with your own match data)
    example_match = {
        'map': 'Mirage',
        'ping': 45,
        'kills': 20,
        'assists': 5,
        'deaths': 15,
        'mvps': 3,
        'hs_percent': 40.0,
        'points': 60
    }

    # Option to use demo data or input own data
    print("1. Use example match data")
    print("2. Input your own match data")
    choice = input("Select an option (1/2): ")

    if choice == '2':
        maps = ['Mirage', 'Dust2', 'Inferno', 'Nuke', 'Overpass', 'Train', 'Vertigo', 'Cache', 'Cobblestone', 'Ancient']
        print("\nAvailable maps:", ', '.join(maps))

        match_data = {}
        match_data['map'] = input("Map name: ")
        match_data['ping'] = int(input("Ping: "))
        match_data['kills'] = int(input("Kills: "))
        match_data['assists'] = int(input("Assists: "))
        match_data['deaths'] = int(input("Deaths: "))
        match_data['mvps'] = int(input("MVPs: "))
        match_data['hs_percent'] = float(input("Headshot percentage: "))
        match_data['points'] = int(input("Points: "))
    else:
        match_data = example_match
        print("\nUsing example match data:")
        for key, value in match_data.items():
            print(f"- {key}: {value}")

    # Prepare the data
    print("\nPreparing match data...")
    match_df = prepare_match_data(match_data)

    # Make prediction
    print("\nPredicting match result...")
    result, probs = predict_match_result(model, match_df)

    # Explain prediction
    explanation = explain_prediction(match_data, result, probs)

    # Display results
    print("\n===== Prediction Results =====")
    print(explanation)
    print("\n==============================")


if __name__ == "__main__":
    main()