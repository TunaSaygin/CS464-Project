import random
import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import cdist
def o1(x_prime, y_prime)->float:
    """
    Objective to minimize the distance between the prediction of x_prime and the desired prediction y_prime.
    """
    if x_prime["prediction"] == y_prime:
        return 0
    else:
        return abs(x_prime["prediction"] - y_prime)

def delta_G(xj, x_prime_j, Rj)->float: #since all our types are numeric in features
    return abs(xj - x_prime_j) / Rj

def o2(x, x_prime, R_hat)->float:
    """
    Objective to quantify the Gower distance between x_prime and x.
    """
    # 
    # print(f"\n\nx[0] = {x[0]},\n x_prime={x_prime} and \n R_hat = {R_hat}")
    if(type(x_prime) is dict):
        return sum(
        delta_G(x[0][j], x_prime['features'][j], R_hat[j])
        for j in range(len(x[0]))
         ) / len(x[0])
    else:
        gower_distance = sum(
            delta_G(x[0][j], x_prime[j], R_hat[j])
            for j in range(len(x[0]))
        ) / len(x[0])
        return gower_distance

def o3(x, x_prime)->float:
    """
    Objective to count the number of features changed (L0 norm).
    """
    # print(f"x[0] = {x[0]}, x_prime={x_prime}")
    if(type(x_prime) is dict):
        return sum(1 for j in range(len(x[0])) if x[0][j] != x_prime["features"][j])
    return sum(1 for j in range(len(x[0])) if x[0][j] != x_prime[j])

def o4(x_prime, x_obs, R_hat)->float:
    """
    Objective to measure the average Gower distance between x_prime and the nearest observed data point.
    """
    # try:
    x_closest = x_obs
    # print(f"\n\n\nx_obs={x_obs}\n\nx_clos={x_closest}")
    if(type(x_obs[0]) is np.ndarray):
        x_closest = find_closest_observed(x_prime, x_obs, R_hat)
    if(type(x_prime) is dict):
        x_prime = x_prime["features"]
    return sum(
        delta_G(x_prime[j], x_closest[j], R_hat[j])
        for j in range(len(x_prime))
    ) / len(x_prime)
    # except:

    #     print(f"\n\n\nx_obs={x_obs}\n\nx_clos={x_closest[0]}\n type x_obs[0] = {str(type(x_closest[0]))}")
    #     sys.exit(1)

def find_closest_observed(x_prime, x_obs, R_hat):
    """
    Find the closest observed data point to x_prime.
    """
    min_distance = float('inf')
    x_closest = None
    # print(f"x_obs = {x_obs}")
    # print(f"x_prime = {x_prime}")
    if(type(x_prime) is dict):
        x_prime = x_prime["features"]
    for x_obs in x_obs:  # Assuming x_obs is a list of lists or a similar iterable of observed data points
        distance = sum(
            delta_G(x_prime[j], x_obs[j], R_hat[j])
            for j in range(len(x_prime))
        ) / len(x_prime)
        if distance < min_distance:
            min_distance = distance
            x_closest = x_obs
    return x_closest

def dominates(x1,x2,x_observational,x_original,model_predict,y_prime,R_hat)->bool:
    # print(f"o1(x1,y_prime, model_predict)={o1(x1,y_prime, model_predict)}")
    # print(f"o2(x_original,x1,R_hat)={o2(x_original,x1,R_hat)}")
    # print(f"o3(x_original,x1) = { o3(x_original,x1)}")
    # print(f"o4(x1,x_observational,R_hat)={o4(x1,x_observational,R_hat)}")
    if x1["o1"]<=x2["o1"] and x1["o2"]<=x2["o2"] and x1["o3"]<=x2["o3"] and x1["o4"] <= x2["o4"]:
        return True
    # print(f"Cannot dominated x1={x1}\n\nx2={x2}")
    return False

def nonDominatedSorting(population,x_observational,x_original, model_predict, y_prime, R_hat):
    print("Before dominated sorted")
    P = population  # The main population
    fronts = [[]]  # The first front is initialized as empty

    for p in P:
        # print(p)
        p['Sp'] = []  # Initialize the set of individuals that p dominates
        p['np'] = 0  # Initialize the domination counter for p

        for q in P:
            if q is not p and dominates(p, q,x_observational,x_original,model_predict,y_prime,R_hat):
                # If p dominates q, add q to the set Sp
                # print("entered")
                p['Sp'].append(q)
            elif q is not p and  dominates(q, p,x_observational,x_original,model_predict,y_prime,R_hat):
                # If q dominates p, increment p's domination counter
                # print("entered")
                p['np'] += 1

        if p['np'] == 0:
            # If p is not dominated by any individual, it belongs to the first front
            p['rank'] = 1
            fronts[0].append(p)
        # else:
        #     # print(f"p['np'] = {p['np']}")
    print("finished first for for loop of nd sort")
    i = 0  # Initialize the front counter
    while fronts[i]:
        Q = []  # The set for storing individuals for the (i+1)th front
        for p in fronts[i]:
            for q in p['Sp']:
                q['np'] -= 1  # Decrement the domination count for q
                if q['np'] == 0:
                    # If q is not dominated by any individual in subsequent fronts
                    q['rank'] = i + 2  # Its rank is set to i+1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    # Remove the last front if it's empty
    if not fronts[-1]:
        fronts.pop()
    print("finished sorting")
    return fronts

def assign_crowding_distance(fronts, y_prime, R_hat, model_predict, x_observational,x_original):
    """
    Assigns crowding distance to each individual in each front.
    
    Args:
        fronts (list): A list of fronts, each front is a list of individuals.
        objectives (list): A list of objective functions.
    """

    # def o1_wrapper(x):
    #     return o1(x,y_prime)
    # def o2_wrapper(x):
    #     return o2(x_original, x, R_hat)
    # def o3_wrapper(x):
    #     return o3(x_original,x)
    # def o4_wrapper(x):
    #     return o4(x,x_observational,R_hat)
    # objectives = [o1_wrapper, o2_wrapper, o3_wrapper, o4_wrapper]
    objectives = ["o1","o2","o3","o4"]
    for front in fronts:
        # Initialize crowding distance for each individual in the front
        for individual in front:
            individual['crowding_distance'] = 0

        # Number of individuals in the front
        n = len(front)

        for m in objectives:
            # Sort the individuals in the front based on the objective m
            front.sort(key=lambda x: x[m])

            # Assign infinite distance to boundary individuals
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')

            # Maximum and minimum values of objective m in the front
            f_max = front[-1][m]
            f_min = front[0][m]

            # Calculate crowding distance for each individual (except boundary individuals)
            for k in range(1, n - 1):
                if(f_max == f_min):
                    front[k]['crowding_distance'] = 0
                else:
                    front[k]['crowding_distance'] += (front[k + 1][m] - front[k - 1][m]) / (f_max - f_min)


def crowded_comparison_operator(individual1, individual2):
    """
    The crowded comparison operator as per NSGA-II.
    It prefers individuals with lower rank (higher non-domination level) or,
    if ranks are equal, individuals with greater crowding distance.
    
    Args:
        individual1 (dict): The first individual.
        individual2 (dict): The second individual.
    
    Returns:
        dict: The winning individual.
    """
    if individual1['rank'] < individual2['rank'] or \
       (individual1['rank'] == individual2['rank'] and individual1['crowding_distance'] > individual2['crowding_distance']):
        return individual1
    else:
        return individual2
    
def crowded_tournament_selection(population, k):
    """
    Performs tournament selection based on crowding distance and non-domination rank.
    
    Args:
        population (list): The population (or a front) with individuals that have 'rank' and 'crowding_distance'.
        k (int): Number of individuals to select.
        
    Returns:
        list: A list of selected individuals.
    """
    selected = []

    while len(selected) < k:
        # Randomly select two individuals from the population for the tournament
        i, j = random.sample(range(len(population)), 2)
        winner = crowded_comparison_operator(population[i], population[j])
        selected.append(winner)

    return selected
def sbx_crossover(p1, p2, eta_c=30):
    p1 = np.array(p1)
    p2 = np.array(p2)
    u = random.random()
    offspring1 = {}
    offspring2 = {}
    offspring1["features"] = np.zeros_like(p1)
    offspring2["features"] = np.zeros_like(p2)
    # print(f"p1={p1}")
    # Perform crossover for each element
    for i in range(len(p1)):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u)**(1 / (eta_c + 1))
        else:
            beta = (1 / (2 * (1 - u)))**(1 / (eta_c + 1))

        c1 = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2 = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        
        # Assign offspring values
        offspring1["features"][i] = c1
        offspring2["features"][i] = c2
    
    return offspring1, offspring2

def polynomial_mutation(child, eta_m, lower_bound, upper_bound):
    """
    Performs polynomial mutation on a child individual.
    
    Args:
        child (np.array): The child individual to mutate.
        eta_m (float): The mutation distribution index.
        lower_bound (np.array): Lower bounds for each feature.
        upper_bound (np.array): Upper bounds for each feature.
        
    Returns:
        np.array: The mutated child individual.
    """
    for k in range(len(child["features"])):
        rk = random.random()
        if rk < 0.5:
            delta_k = (2*rk)**(1/(eta_m+1)) - 1
        else:
            delta_k = 1 - (2*(1-rk))**(1/(eta_m+1))
        
        # Mutation operation
        # print(f"child = {child}")
        child["features"][k] = child["features"][k] + (upper_bound[k] - lower_bound[k]) * delta_k
        
        # Ensure that the mutated features is within bounds
        child["features"][k] = min(max(child["features"][k], lower_bound[k]), upper_bound[k])
    
    return child

def generate_offspring(selected_individuals, eta_c, eta_m, lower_bound, upper_bound, population_size):
    new_generation = []
    
    # Ensure we have an even number of individuals for pairing
    if len(selected_individuals) % 2 != 0:
        selected_individuals.append(random.choice(selected_individuals))
    
    # Shuffle to ensure random pairing
    random.shuffle(selected_individuals)
    
    while len(new_generation) < population_size:
        for i in range(0, len(selected_individuals), 2):
            parent1, parent2 = selected_individuals[i]["features"], selected_individuals[i+1]["features"]
            # Apply crossover
            offspring1, offspring2 = sbx_crossover(parent1, parent2, eta_c)
            
            # Apply mutation
            offspring1 = polynomial_mutation(offspring1, eta_m, lower_bound, upper_bound)
            offspring2 = polynomial_mutation(offspring2, eta_m, lower_bound, upper_bound)
            
            # Add offspring to the new generation
            new_generation.extend([offspring1, offspring2])
            if len(new_generation) >= population_size:
                break
                
    # Ensure the new generation does not exceed the desired population size
    return new_generation[:population_size]

def get_feature_range(x_observational):
    # Calculate min and max for each feature/column in the NumPy array
    mins = x_observational.min(axis=0)  # Min values for each column/feature
    maxs = x_observational.max(axis=0)  # Max values for each column/feature
    
    # Construct feature_ranges as a dictionary {feature_index: (min_val, max_val)}
    feature_ranges = {i: (min_val, max_val) for i, (min_val, max_val) in enumerate(zip(mins, maxs))}
    return feature_ranges, mins, maxs

# R_hat can be calculated similarly as the difference between max and min for each feature
def calculate_R_hat(x_observational):
    mins = x_observational.min(axis=0)
    maxs = x_observational.max(axis=0)
    R_hat = maxs - mins
    return R_hat
def generate_random_individual(feature_ranges):
    individual = {'features': np.array([np.random.uniform(low, high) for feature, (low, high) in feature_ranges.items()]),
                  'np': 0, 'Sp': [], 'crowding_distance': 0}
    return individual
def generate_population(population_size, feature_ranges):
    population = []
    for _ in range(population_size):
        population.append(generate_random_individual(feature_ranges))
    return population
def prepare_batch(population):
    # print("First individual in population:", population[0])
    """Prepare a batch from the population's features for model prediction."""
    # Extract features from each individual and stack them into a single NumPy array
    # features_batch = np.array([ind["features"] for ind in population])
    features_batch = np.array([ind.get("features", np.array([])) for ind in population])
    return features_batch

def assign_predictions(population, model_predict):
    """Perform batch prediction and assign the predictions to individuals."""
    features_batch = prepare_batch(population)
    predictions = model_predict(features_batch)
    
    # Assign predictions back to the individuals
    for ind, prediction in zip(population, predictions):
        ind["prediction"] = prediction

def find_closest_observed_vectorized(x_primes, x_obs):
    """
    Find the closest observed data point for each x_prime using vectorized operations.
    
    Args:
        x_primes (np.array): A 2D NumPy array of shape (n_samples, n_features) containing the feature vectors.
        x_obs (np.array): A 2D NumPy array of shape (m_samples, n_features) containing the observed data points.
        
    Returns:
        np.array: A 2D NumPy array containing the closest observed data points for each x_prime.
    """
    # Compute the pairwise Euclidean distances between x_primes and x_obs
    distances = cdist(x_primes, x_obs, metric='euclidean')
    
    # Find the index of the minimum distance for each x_prime
    min_indices = np.argmin(distances, axis=1)
    
    # Select the closest observed data points based on the indices
    closest_observed = x_obs[min_indices]
    
    return closest_observed

def compute_objectives(population, x_observational, x_original, y_prime, R_hat):
    # Assuming x_observational and x_original are preprocessed as needed for the calculations
    x_obs_closest = find_closest_observed_vectorized([ind["features"] for ind in population], x_observational)
    
    for ind, x_closest in zip(population, x_obs_closest):
        # Use precomputed prediction for o1
        ind["o1"] = o1(ind, y_prime)
        # print(f"\n\n\nx_original = {x_original}")
        # Compute o2, o3, o4 values
        ind["o2"] = o2(x_original, ind["features"], R_hat)
        ind["o3"] = o3(x_original, ind["features"])
        ind["o4"] = o4(ind["features"], x_closest, R_hat)

def create_counterfactuals(x_original, x_observational, y_target, model_predict,generations=50, population_count=100):
    feature_ranges, mins, maxs = get_feature_range(x_observational)
    R_hat = calculate_R_hat(x_observational)
    population = generate_population(population_count,feature_ranges)
    # print(population)
    for i in range(generations):
        assign_predictions(population,model_predict)
        compute_objectives(population,x_observational,x_original,y_target,R_hat)
        fronts = nonDominatedSorting(population,x_observational,x_original,model_predict,y_target,R_hat)
        assign_crowding_distance(fronts, y_target,R_hat,model_predict,x_observational,x_original)
        survived = crowded_tournament_selection(population,population_count/2)
        population = generate_offspring(survived,eta_c=20,eta_m=20,lower_bound=mins,upper_bound=maxs,population_size=population_count)
        print(f"generation {i} Created")
    
    for i, individual in enumerate(population):
        if(i<10):
            print(individual)
        else:
            break
    return population