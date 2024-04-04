from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
#lets implement our NSGA-II algorithm  deap
def find_closest_observed(x_prime, X_obs, R_hat):
    """
    Find the closest observed data point to x_prime.

    Args:
    x_prime: The counterfactual instance.
    X_obs: The dataset of observed instances.
    R_hat: The range of values for each feature, used to normalize distances.

    Returns:
    The closest observed data point to x_prime.
    """
    min_distance = float('inf')
    x_closest = None
    # print(f"X_obs={X_obs}")
    for x_obs in X_obs:
        distance = sum(
            (1/R_hat[j] * abs(x_prime[j] - x_obs[j]))
            for j in range(len(x_prime))
        ) / len(x_prime)
        if distance < min_distance:
            min_distance = distance
            x_closest = x_obs
    return x_closest

import random
import numpy as np

def sbx_crossover(parent1, parent2, eta_c=20, low=0, up=1):
    """Perform Simulated Binary Crossover (SBX) on two parents.
    
    Args:
        parent1 (list): First parent for crossover.
        parent2 (list): Second parent for crossover.
        eta_c (int): Distribution index for crossover.
        low (float): Lower bound of the decision variables.
        up (float): Upper bound of the decision variables.
    
    Returns:
        tuple: Two offspring in the form of a tuple.
    """
    size = min(len(parent1), len(parent2))
    child1, child2 = parent1[:], parent2[:]
    for i in range(size):
        if random.random() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:  # Avoid division by zero
                mean = 0.5 * (parent1[i] + parent2[i])
                beta = 1.0 + (2.0 * (mean - low) / (parent2[i] - parent1[i]))
                alpha = 2.0 - beta**-(eta_c + 1)
                rand = random.random()
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta_c + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta_c + 1))
                child1[i] = 0.5 * ((1 + beta_q) * parent1[i] + (1 - beta_q) * parent2[i])

                beta = 1.0 + (2.0 * (up - mean) / (parent2[i] - parent1[i]))
                alpha = 2.0 - beta**-(eta_c + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta_c + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta_c + 1))
                child2[i] = 0.5 * ((1 - beta_q) * parent1[i] + (1 + beta_q) * parent2[i])
    return child1, child2

def polynomial_mutation(individual, eta_m=20, low=0, up=1):
    """Perform polynomial mutation on an individual.
    
    Args:
        individual (list): The individual to mutate.
        eta_m (int): Distribution index for mutation.
        low (float): Lower bound of the decision variables.
        up (float): Upper bound of the decision variables.
    
    Returns:
        list: A mutated individual.
    """
    size = len(individual)
    for i in range(size):
        if random.random() <= 0.5:
            u = random.random()
            delta = 0
            if u < 0.5:
                delta = (2*u)**(1/(eta_m+1)) - 1
            else:
                delta = 1 - (2*(1-u))**(1/(eta_m+1))
            individual[i] += delta
            individual[i] = min(max(individual[i], low), up)
    return individual


def o1(x_prime, y_prime, model_predict):
    column_names = [
    'group', 'ID', 'age', 'gendera', 'BMI', 'hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes',
    'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'heart rate',
    'Systolic blood pressure', 'Diastolic blood pressure', 'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
    'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW', 'Leucocyte', 'Platelets', 'Neutrophils', 'Basophils',
    'Lymphocyte', 'PT', 'INR', 'NT-proBNP', 'Creatine kinase', 'Creatinine', 'Urea nitrogen', 'glucose',
    'Blood potassium', 'Blood sodium', 'Blood calcium', 'Chloride', 'Anion gap', 'Magnesium ion', 'PH', 'Bicarbonate',
    'Lactic acid', 'PCO2', 'EF'
    ]
    x_prime_array = np.array(x_prime)
    # Convert x_prime to a DataFrame and reshape to 2D if needed
    # Note: x_prime is expected to be a 1D numpy array here
    x_prime_df = pd.DataFrame(x_prime_array.reshape(1, -1), columns=column_names)
    # print(f"X prime type is {type(x_prime_df)}")
    prediction = model_predict(x_prime_df)
    print(f"prediction={prediction}")
    if prediction[0] == y_prime:
        print(f"o1=0")
        return 0
    else:
        print(f"o1={abs(prediction-y_prime)}")
        return abs(prediction - y_prime)
def o2(x, x_prime, R_hat):
    # print(f"x and x_prime type is {str(type(x))} and {str(type(x_prime))} respectively")
    # print(f"x[0]={x[0]} and len(x) = {len(x)}\n\n----------\n")
    # print(f"x_prime={x_prime}")
    gower_distance = sum(
    (1/R_hat[j] * abs(x[j] - x_prime[j]))
    for j in range(len(x))
    )
    return gower_distance / len(x)
def o3(x, x_prime):
    print(f"o3={sum(1 for j in range(len(x)) if x[j] != x_prime[j])}")
    return sum(1 for j in range(len(x)) if x[j] != x_prime[j])

def o4(x_prime, X_obs, R_hat):
    x_closest = find_closest_observed(x_prime, X_obs,R_hat)
    gower_distance = sum(
        (1/R_hat[j] * abs(x_prime[j] - x_closest[j]))
        for j in range(len(x_prime))
    )
    print(f"gower_distance{gower_distance}")
    return gower_distance / len(x_prime)
def generate_eval_func(x_original, x_observational, desired_outcome,model_predict):
    def evaluate_individual(individual):
        # `individual` is a counterfactual x'
        # You need to define how to get original instance x and desired outcome y' and observational data X_obs
        # R_hat is the observed value range for each feature
        # The function `numeric` should return True if the feature is numerical, and `model_predict` is your prediction model
        R_hat = [ x_observational[col].max() - x_observational[col].min() for col in x_observational ]
        x = x_original.iloc[0].to_numpy()
        y_prime = desired_outcome
        X_obs = x_observational.to_numpy()
        # R_hat = ... # range of values for each feature
        
        return (
            o1(individual, y_prime, model_predict),
            o2(x, individual, R_hat),
            o3(x, individual),
            o4(individual, X_obs, R_hat)
        )
    return evaluate_individual
# Create the fitness and individual classes
def create_population(x_original, x_observational, desired_outcome,model_predict):
    IND_SIZE = 50
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))  # Minimize all objectives
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    # Set up the toolbox
    toolbox = base.Toolbox()
    print(f"type:{str(type(x_observational))}")
    # feature_ranges =  [ (x_observational[col].max() , x_observational[col].min()) for col in x_observational ]
    # Assuming `x_observational` is a pandas DataFrame
    feature_ranges = pd.DataFrame(index=x_observational.columns)
    feature_ranges['min'] = x_observational.min()
    feature_ranges['max'] = x_observational.max()
    # Identify binary features as those with a min of 0 and a max of 1
    feature_ranges['is_binary'] = (feature_ranges['min'] == 0) & (feature_ranges['max'] == 1)

    def custom_attr_generator(feature_ranges):
        """Generates a value for each feature based on its characteristics (binary or numeric range).
        
        Args:
            feature_ranges (pd.DataFrame): A DataFrame with min, max, and is_binary for each feature.
        
        Returns:
            A generator function that when called returns a list of values for an individual.
        """
        def generate():
            values = []
            for _, row in feature_ranges.iterrows():
                if row['is_binary']:
                    value = random.randint(0, 1)  # Generate a binary value
                else:
                    value = random.uniform(row['min'], row['max'])  # Generate a value within the numeric range
                values.append(value)
            return values
        return generate    
    # Attribute generator
    # I would like to make it as realistic as possible 
    # toolbox.register("attr_float", random.uniform, 1, 100)  # Adjust bounds as needed
    # Register the custom attribute generator
    toolbox.register("attr_custom", custom_attr_generator(feature_ranges=feature_ranges))
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_custom, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", generate_eval_func(x_original, x_observational, desired_outcome,model_predict))  # You need to implement this function
    # Register mating and mutation operators
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=100, eta=20)  # Example bounds and eta
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=100, eta=20, indpb=0.1)  # Example bounds, eta, and indpb
    toolbox.register("select", tools.selNSGA2)

    # ... (other toolbox registrations like mutation, crossover, and selection)

    # Create initial population and run the NSGA-II algorithm
    population = toolbox.population(n=100)
    halloffame = tools.HallOfFame(10)  # Adjust the size as needed
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    # algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=50, stats=stats, halloffame=halloffame)

    final_population = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=50, stats=stats, halloffame=halloffame, verbose=True)

    return final_population, halloffame