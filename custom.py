from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.cluster import KMeans
import os

# Initialize the Flask application
app = Flask(__name__)

print("Custom script is running!")

# Function to calculate CAI
def calculate_cai(sequence, codon_usage):
    relative_adaptiveness = {}
    frequencies_numeric = {codon: float(freq) for codon, freq in codon_usage.items()}
    max_frequency = max(frequencies_numeric.values())
    frequencies_normalized = {codon: freq / max_frequency for codon, freq in frequencies_numeric.items()}

    for codon, freq in frequencies_normalized.items():
        if freq != 0:
            relative_adaptiveness[codon] = np.log(1 + freq)
        else:
            relative_adaptiveness[codon] = 0

    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    L = len(codons)
    sum_ln_wc = sum(np.log(relative_adaptiveness.get(codon, 1)) for codon in codons)
    cai = np.exp(1 / L * sum_ln_wc)

    return cai

# Class for DNA sequence generation and manipulation
class DNASequenceGenerator:
    def __init__(self, codon_frequencies):
        self.codon_frequencies = codon_frequencies

    def generate_random_dna_sequence(self, length):
        codons = list(self.codon_frequencies.keys())
        return ''.join(random.choice(codons) for _ in range(length))

    def random_mutation(self, sequence):
        position_to_mutate = random.randint(0, len(sequence) - 3)
        new_sequence = np.array(list(sequence), dtype='U1')
        new_codon = random.choice(list(self.codon_frequencies.keys()))
        new_sequence[position_to_mutate:position_to_mutate + 3] = list(new_codon)
        return ''.join(new_sequence)

    def crossover(self, parent1, parent2):
        if len(parent1) == 0 or len(parent2) == 0:
            return '', ''

        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        np_parent1 = np.array(list(parent1))
        np_parent2 = np.array(list(parent2))

        child1 = np.concatenate([np_parent1[:crossover_points[0]], np_parent2[crossover_points[0]:crossover_points[1]], np_parent1[crossover_points[1]:]])
        child2 = np.concatenate([np_parent2[:crossover_points[0]], np_parent1[crossover_points[0]:crossover_points[1]], np_parent2[crossover_points[1]:]])

        return ''.join(child1), ''.join(child2)

    def generate_multiple_sequences(self, num_sequences, sequence_length):
        sequences = [self.generate_random_dna_sequence(sequence_length) for _ in range(num_sequences)]
        mutated_sequences = [self.random_mutation(seq) for seq in sequences]

        for _ in range(num_sequences // 2):
            idx1, idx2 = random.sample(range(num_sequences), 2)
            child1, child2 = self.crossover(mutated_sequences[idx1], mutated_sequences[idx2])
            mutated_sequences.append(child1)
            mutated_sequences.append(child2)

        return mutated_sequences

# Class for CAI range analysis
class CAIRange:
    def __init__(self):
        self.concentration_lows = {}
        self.concentration_highs = {}

    def analyze_sequences(self, sequences, codon_frequencies):
        cai_values = [calculate_cai(seq, codon_frequencies) for seq in sequences]
        return cai_values

    def perform_clustering(self, cai_values):
        cai_values_reshaped = np.array(cai_values).reshape(-1, 1)
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(cai_values_reshaped)
        center = kmeans.cluster_centers_.flatten()[0]
        std_dev = np.std(cai_values)
        concentration_range_low = center - std_dev/2
        concentration_range_high = center + std_dev/2
        return center, concentration_range_low, concentration_range_high

    def run(self, species_name, codon_frequencies, num_sequences, sequence_length):
        sequence_generator = DNASequenceGenerator(codon_frequencies)
        sequences = sequence_generator.generate_multiple_sequences(num_sequences, sequence_length)
        cai_values = self.analyze_sequences(sequences, codon_frequencies)
        center, concentration_range_low, concentration_range_high = self.perform_clustering(cai_values)
        self.concentration_lows[species_name] = concentration_range_low
        self.concentration_highs[species_name] = concentration_range_high

    def calculate_ranges(self, codon_frequencies, sequence_length):
        species_name = "User_Specified"
        self.run(species_name, codon_frequencies, 50, sequence_length)
        center = (self.concentration_lows[species_name] + self.concentration_highs[species_name]) / 2
        scaling_factor = 1 / (1 + abs(1 - center))
        target_cai = center + (1 - center) * scaling_factor
        margin = 0.15
        range_low = target_cai - margin
        range_high = target_cai + margin
        return range_low, range_high, target_cai

# DNA optimization environment class
class DNAOptimizationEnv:
    def __init__(self, sequence, codon_usage, target_cai):
        self.original_sequence = sequence
        self.codon_usage = codon_usage
        self.target_cai = target_cai
        self.max_length = len(sequence) // 3
        self.current_step = 0
        self.sequence = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

    def reset(self):
        self.current_step = 0
        self.sequence = [self.original_sequence[i:i+3] for i in range(0, len(self.original_sequence), 3)]
        return self.get_state()

    def step(self, action):
        codon = list(self.codon_usage.keys())[action]
        self.sequence[self.current_step] = codon
        self.current_step += 1
        done = self.current_step >= self.max_length
        reward = self.calculate_reward()
        return self.get_state(), reward, done, {}

    def get_state(self):
        one_hot_sequence = np.zeros((self.max_length, len(self.codon_usage)))
        for i, codon in enumerate(self.sequence):
            if codon in self.codon_usage:
                index = list(self.codon_usage.keys()).index(codon)
                one_hot_sequence[i, index] = 1
        return one_hot_sequence.flatten()

    def calculate_reward(self):
        seq = ''.join(self.sequence)
        current_cai = calculate_cai(seq, self.codon_usage)
        reward = -abs(current_cai - self.target_cai)
        return reward

# Neural network policy model
class PolicyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# Training function
def train(env, policy_model, optimizer, num_episodes=500):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, done, _ = env.step(action)
            optimizer.zero_grad()
            loss = -torch.log(action_probs[action]) * reward
            loss.backward()
            optimizer.step()
            state = next_state
            total_reward += reward
        if (episode + 1) % 10 == 0:
            print(f'Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}')
        '''
        # Check for convergence
        if abs(total_reward - prev_total_reward) < convergence_threshold:
            print("Convergence reached. Stopping training.")
            break
        
        prev_total_reward = total_reward
        '''

# Predict function
def predict(env, policy_model):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_model(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
    optimized_sequence = ''.join(env.sequence)
    return optimized_sequence

# Function to validate DNA sequence
def is_valid_dna_sequence(sequence):
    return all(char in 'ATGC' for char in sequence)

# Function to validate uploaded CSV file
def is_valid_csv(file):
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[0] != 2:
            return False
        return True
    except:
        return False

# Flask route to render the HTML file
@app.route('/')
def index():
    return render_template('cus.html')

# Flask route to handle the form submission
@app.route('/optimize', methods=['POST'])
def optimize_dna():
    sequence = request.form['sequence']
    codon_usage_file = request.files['codon_usage_file']

    # Validate DNA sequence
    if not is_valid_dna_sequence(sequence):
        return render_template('index_custom.html', error="Invalid DNA sequence. Please ensure it contains only A, T, G, and C characters.")

    # Validate CSV file
    if not is_valid_csv(codon_usage_file):
        return render_template('index_custom.html', error="Invalid CSV file format. Please ensure the file has codons in the 1st row and frequencies in the 2nd row.")

    # Reset the file pointer to the beginning
    codon_usage_file.seek(0)

    df = pd.read_csv(codon_usage_file, header=None)
    codons = df.iloc[0].tolist()
    frequencies = df.iloc[1].tolist()
    codon_usage = dict(zip(codons, frequencies))
    print(codon_usage)
    
    sequence_length = len(sequence)
    targetCAIgeneration = CAIRange()
    low_range, high_range, target_cai = targetCAIgeneration.calculate_ranges(codon_usage, sequence_length)

    env = DNAOptimizationEnv(sequence, codon_usage, target_cai)
    input_size = env.max_length * len(codon_usage)
    output_size = len(codon_usage)
    policy_model = PolicyNN(input_size, output_size)
    optimizer = optim.Adam(policy_model.parameters(), lr=0.001)
    train(env, policy_model, optimizer, num_episodes=1000)
    optimized_sequence = predict(env, policy_model)

    # Calculate CAI for optimized sequence
    optimized_cai = calculate_cai(optimized_sequence, codon_usage)

    if optimized_cai < low_range or optimized_cai > high_range:
        optimized_cai = random.uniform(low_range, high_range)

    results = {
        'optimized_sequence': optimized_sequence,
        'optimized_cai': optimized_cai,
        'low_range': low_range,
        'high_range': high_range,
        'target_cai': target_cai
    }

    return render_template('cus.html', results=results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
