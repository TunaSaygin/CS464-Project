import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
    # print(f"\n\nx = {x},\n x_prime={x_prime} and \n R_hat = {R_hat}")
    if(type(x_prime) is dict):
        return sum(
        delta_G(x[j], x_prime['features'][j], R_hat[j])
        for j in range(len(x))
         ) / len(x)
    else:
        gower_distance = sum(
            delta_G(x[j], x_prime[j], R_hat[j])
            for j in range(len(x))
        ) / len(x)
        # print("gower_distance = ", gower_distance)
        return gower_distance

def o3(x, x_prime)->float:
    """
    Objective to count the number of features changed (L0 norm).
    """
    # print(f"x[0] = {x[0]}, x_prime={x_prime}")
    if(type(x_prime) is dict):
        return sum(1 for j in range(len(x)) if x[j] != x_prime["features"][j])
    return sum(1 for j in range(len(x)) if x[j] != x_prime[j])

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
    # print("Before dominated sorted")
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
    # print("finished first for for loop of nd sort")
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
    # print("finished sorting")
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
    
def sort_population_by_objectives(population, target_outcome):
    # Sort population based on the absolute difference from the target for o1 and then by o2
    return sorted(population, key=lambda x: (abs(x['o1'] - target_outcome), x['o2']))
def crowded_tournament_selection(population, k, target_outcome):
    """
    Performs tournament selection based on crowding distance and non-domination rank.
    
    Args:
        population (list): The population (or a front) with individuals that have 'rank' and 'crowding_distance'.
        k (int): Number of individuals to select.
        
    Returns:
        list: A list of selected individuals.
    """
    selected = []
    sorted_population = sort_population_by_objectives(population,target_outcome)
    weights = [1 / (i + 1) for i in range(len(sorted_population))]  # Higher weight to lower index (better rank)
    while len(selected) < k:
        # Randomly select two individuals from the population for the tournament
        i, j = random.choices(range(len(sorted_population)), weights=weights, k=2)
        winner = crowded_comparison_operator(sorted_population[i], sorted_population[j])
        selected.append(winner)

    return selected
def sbx_crossover(p1, p2, eta_c=30,feature_ranges=None):
    p1, mutable1 = np.array(p1["features"]), p1["mutable_features"]
    p2, mutable2 = np.array(p2["features"]), p2["mutable_features"]
    p1 = np.array(p1)
    p2 = np.array(p2)
    u = random.random()
    offspring1 = {"features": p1, "mutable_features": mutable1}
    offspring2 = {"features": p2, "mutable_features": mutable2}
    # print(f"p1={p1}")
    # Perform crossover for each element
    for i in range(len(p1)):
        if (mutable1[i] or mutable2[i]) and random.random()>0.9:
            offspring1["mutable_features"][i] = offspring2["mutable_features"][i] = True
            if feature_ranges[i] == (0, 1):
                # Binary feature, pick randomly from parents
                offspring1["features"][i] = random.choice([p1[i], p2[i]])
                offspring2["features"][i] = random.choice([p1[i], p2[i]])
            else:
                # Perform standard SBX crossover
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u)**(1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u)))**(1 / (eta_c + 1))

                c1 = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2 = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                
                offspring1["features"][i] = c1
                offspring2["features"][i] = c2
    
    return offspring1, offspring2

def polynomial_mutation(child, eta_m, lower_bound, upper_bound, feature_ranges):
    """
    Performs adaptive polynomial mutation on a child individual.

    Args:
        child (dict): The child individual to mutate, with 'features' key containing the array.
        eta_m (float): The mutation distribution index.
        lower_bound (np.array): Lower bounds for each feature.
        upper_bound (np.array): Upper bounds for each feature.
        feature_ranges (list): List of tuples (low, high) for each feature.

    Returns:
        dict: The mutated child individual.
    """
    for k in range(len(child["features"])):
        # Calculate the median of the feature range
        median = (upper_bound[k] + lower_bound[k]) / 2
        if child["mutable_features"][k]:  # Check if feature is mutable
            if feature_ranges[k] == (0, 1):
                # Binary feature, flip based on mutation probability
                if random.random() < 0.1:  # Example mutation probability
                    child["features"][k] = 1 - child["features"][k]
            else:
                # Perform standard polynomial mutation
                rk = random.random()
                if rk < 0.5:
                    delta_k = (2 * rk)**(1 / (eta_m + 1)) - 1
                else:
                    delta_k = 1 - (2 * (1 - rk))**(1 / (eta_m + 1))

                # Determine mutation step size based on proximity to the median
                step_size = (upper_bound[k] - lower_bound[k]) * delta_k
                distance_to_median = abs(child["features"][k] - median)
                # Scale the mutation step size based on distance to median (adaptive component)
                step_size *= (1 - (distance_to_median / (upper_bound[k] - lower_bound[k])))

                child["features"][k] += step_size
                # Ensure mutated feature remains within bounds
                child["features"][k] = np.clip(child["features"][k], lower_bound[k], upper_bound[k])

    return child

def generate_offspring(selected_individuals, eta_c, eta_m, lower_bound, upper_bound, population_size,feature_ranges):
    new_generation = []
    
    # Ensure we have an even number of individuals for pairing
    if len(selected_individuals) % 2 != 0:
        selected_individuals.append(random.choice(selected_individuals))
    
    # Shuffle to ensure random pairing
    random.shuffle(selected_individuals)
    
    while len(new_generation) < population_size:
        for i in range(0, len(selected_individuals), 2):
            parent1, parent2 = selected_individuals[i], selected_individuals[i+1]
            # Apply crossover
            offspring1, offspring2 = sbx_crossover(parent1, parent2, eta_c,feature_ranges)
            
            # Apply mutation
            offspring1 = polynomial_mutation(offspring1, eta_m, lower_bound, upper_bound,feature_ranges)
            offspring2 = polynomial_mutation(offspring2, eta_m, lower_bound, upper_bound,feature_ranges)
            
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
    individual = {'features': np.zeros(len(feature_ranges))}
    for i, ((low, high)) in enumerate(feature_ranges.values()):
        if low == 0 and high == 1:
            # If the range is 0 to 1, assign randomly either 0 or 1
            individual['features'][i] = random.choice([0, 1])
        else:
            # Otherwise, assign a random value within the range
            individual['features'][i] = np.random.uniform(low, high)
    individual.update({'np': 0, 'Sp': [], 'crowding_distance': 0})
    return individual

def generate_seeded_individual(feature_ranges, actual_data_point):
    # print("Acutal datapoint: ",actual_data_point)
    num_features_to_change = random.randint(2, 4)  # Randomly determine number of features to change
    individual = {
                    'features': np.zeros(len(feature_ranges)),
                    'mutable_features': [False] * len(feature_ranges)
                }
    features_to_change = random.sample(list(feature_ranges.keys()), num_features_to_change)
    for i, key in enumerate(feature_ranges):
        (low, high) = feature_ranges[key]
        if key in features_to_change:
            if low == 0 and high == 1:
                individual['features'][i] = actual_data_point[i]
            else:
                perturbation = (high - low) * 0.2  # Adjust the perturbation scale as needed
                individual['features'][i] = np.clip(actual_data_point[i] + np.random.uniform(-perturbation, perturbation), low, high)
            individual['mutable_features'][i] = True
        else:
            individual['features'][i] = actual_data_point[i]  # keep original for unchanged features
    individual.update({'np': 0, 'Sp': [], 'crowding_distance': 0})
    return individual

def generate_population(population_size, feature_ranges, actual_dataset):
    population = []
    # Seed with actual data points
    num_seeds = int(population_size * 1)  # Adjust the proportion as needed
    for _ in range(num_seeds):
        population.append(generate_seeded_individual(feature_ranges, actual_dataset))
    # Generate the rest as random
    for _ in range(population_size - num_seeds):
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
        # print("x_original",x_original)
        ind["o2"] = o2(x_original, ind["features"], R_hat)
        ind["o3"] = o3(x_original, ind["features"])
        ind["o4"] = o4(ind["features"], x_closest, R_hat)

def calculate_percentage_change(original, counterfactual):
    percent_changes = np.zeros_like(original, dtype=float)
    
    for i, (orig, cf) in enumerate(zip(original, counterfactual)):
        if orig == 0 and cf != 0:
            percent_changes[i] = 100  # From 0 to non-zero
        elif orig != 0 and cf == 0:
            percent_changes[i] = 100  # From non-zero to 0
        elif orig == 0 and cf == 0:
            percent_changes[i] = 0   # Both are zero, no change
        else:
            percent_changes[i] = ((cf - orig) / orig) * 100  # Normal percentage change calculation
        
    return percent_changes
def filter_individuals_by_feature_change(population, original_individual):
    filtered_population = []
    for individual in population:
        percent_changes = calculate_percentage_change(np.array(original_individual), np.array(individual['features']))
        # Apply any additional filtering criteria based on percent_changes
        if np.all((-100 <= percent_changes) & (percent_changes <= 100)):
            filtered_population.append(individual)
    return filtered_population
def create_counterfactuals(x_original, x_observational, y_target, model_predict,generations=50, population_count=100):
    feature_ranges, mins, maxs = get_feature_range(x_observational)
    R_hat = calculate_R_hat(x_observational)
    population = generate_population(population_count,feature_ranges,x_original)
    # print(x_original)
    for i in range(generations):
        print(f"Generation {i}")
        assign_predictions(population,model_predict)
        compute_objectives(population,x_observational,x_original,y_target,R_hat)
        fronts = nonDominatedSorting(population,x_observational,x_original,model_predict,y_target,R_hat)
        assign_crowding_distance(fronts, y_target,R_hat,model_predict,x_observational,x_original)
        survived = crowded_tournament_selection(population,population_count/2,y_target)
        population = generate_offspring(survived,eta_c=20,eta_m=20,lower_bound=mins,upper_bound=maxs,population_size=population_count,feature_ranges=feature_ranges)
        # print(f"generation {i} Created")
    #assign predictions for the final generation
    assign_predictions(population,model_predict)
    compute_objectives(population,x_observational,x_original,y_target,R_hat)
    population_sorted = sorted(population, key=lambda x: x['o2'])
    population_sorted = filter_individuals_by_feature_change(population_sorted,x_original)
    distilled_population = []
    individual_count = 0
    for j, individual in enumerate(population_sorted):
        if(individual['prediction'] == y_target):
            distilled_population.append(individual)
            individual_count +=1
        if individual_count >10:
            break
    print("Not enough individuals are present. Please increase the generation")
    while(len(distilled_population) == 0):
        print(f"Additional Generation {i}")
        i+=1
        fronts = nonDominatedSorting(population,x_observational,x_original,model_predict,y_target,R_hat)
        assign_crowding_distance(fronts, y_target,R_hat,model_predict,x_observational,x_original)
        survived = crowded_tournament_selection(population,population_count/2,y_target)
        population = generate_offspring(survived,eta_c=20,eta_m=20,lower_bound=mins,upper_bound=maxs,population_size=population_count,feature_ranges=feature_ranges)
        assign_predictions(population,model_predict)
        compute_objectives(population,x_observational,x_original,y_target,R_hat)
        population_sorted = sorted(population, key=lambda x: x['o2'])
        for j, individual in enumerate(population_sorted):
            if(individual['prediction'] == y_target):
                distilled_population.append(individual)
                individual_count +=1
            if individual_count >10:
                break
    return distilled_population

def plot_feature_changes(df):
    fig = go.Figure()
    # Iterate only over rows where there's a significant change
    for index, row in df.iterrows():
        if abs(row['PercentChange']) > 0:  # Adjust this threshold if needed
            fig.add_trace(go.Bar(
                x=[f"{index} Original", f"{index} Counterfactual"],
                y=[row['Original'], row['Counterfactual']],
                name=index
            ))

    fig.update_layout(
        title="Comparison of Original and Counterfactual Values for Significant Changes",
        xaxis_title="Features",
        yaxis_title="Values",
        barmode='group'
    )
    fig.show()

def adjust_counterfactuals(x_original, counterfactual, features, normal_ranges):
    x_original = np.array(x_original)
    counterfactual_p = np.array(counterfactual)
    percent_change = calculate_percentage_change(x_original, counterfactual_p)
    changes_to_normal = []

    for idx, feature in enumerate(features):
        value = x_original[idx]
        range_type = normal_ranges[feature]
        if range_type == -1:  # No valid calculation required
            change_to_normal = 0
        elif range_type == 0:  # Binary feature
            if x_original[idx] == 1:
                change_to_normal = -90  # Need a 90% reduction to reach 0
            else:
                change_to_normal = 0  # No change needed, it's already 0
        else:  # Numerical range
            normal_min, normal_max = range_type
            if value < normal_min:
                change_to_normal = (normal_min - value) / abs(value) * 100 if value != 0 else float('inf')
            elif value > normal_max:
                change_to_normal = (normal_max - value) / abs(value) * 100 if value != 0 else float('inf')
            else:
                change_to_normal = 0  # Already within the normal range
        changes_to_normal.append(change_to_normal)
    
    return percent_change, changes_to_normal
def plot_features(x_original, counterfactual,original_prediction ,counterfactual_prediction ):
    print("counterfactual: ",counterfactual)
    features = [
    # "group", 
    "age", "gendera", "BMI", "hypertensive", "atrialfibrillation", 
    "CHD with no MI", "diabetes", "deficiencyanemias", "depression", "Hyperlipemia", 
    "Renal failure", "COPD", "heart rate", "Systolic blood pressure", 
    "Diastolic blood pressure", "Respiratory rate", "temperature", "SP O2", 
    "Urine output", "hematocrit", "RBC", "MCH", "MCHC", "MCV", "RDW", "Leucocyte", 
    "Platelets", "Neutrophils", "Basophils", "Lymphocyte", "PT", "INR", "NT-proBNP", 
    "Creatine kinase", "Creatinine", "Urea nitrogen", "glucose", "Blood potassium", 
    "Blood sodium", "Blood calcium", "Chloride", "Anion gap", "Magnesium ion", "PH", 
    "Bicarbonate", "Lactic acid", "PCO2", "EF"
    ]
    normal_ranges = {
    "age": -1,  # Age doesn't have a normal range as it varies widely by context
    "gendera": -1,  # Gender is categorical
    "BMI": (18.5, 24.9),
    "hypertensive": 0,  # Typically binary (yes/no)
    "atrialfibrillation": 0,  # Typically binary (yes/no)
    "CHD with no MI": 0,  # Typically binary (yes/no)
    "diabetes": 0,  # Typically binary (yes/no)
    "deficiencyanemias": 0,  # Binary or specific medical test ranges
    "depression": 0,  # Typically binary (yes/no) or specific scale values
    "Hyperlipemia": 0,  # Typically binary (yes/no); lipid profiles have specific ranges
    "Renal failure": 0,  # Binary or specific medical test values
    "COPD": 0,  # Binary (yes/no)
    "heart rate": (60, 100),
    "Systolic blood pressure": (90, 120),
    "Diastolic blood pressure": (60, 80),
    "Respiratory rate": (12, 20),
    "temperature": (36.1, 37.2),
    "SP O2": (95, 100),
    "Urine output": (800, 2000),
    "hematocrit": (40, 52),  # Typical range for men, adjust if needed for women or others
    "RBC": (4.7, 6.1),  # Typical range for men
    "MCH": (27.5, 33.2),
    "MCHC": (33.4, 35.5),
    "MCV": (80, 96),
    "RDW": (11.6, 14.6),
    "Leucocyte": (3.5, 10.5),
    "Platelets": (150, 450),
    "Neutrophils": (1.8, 7.7),
    "Basophils": (0, 0.3),
    "Lymphocyte": (1.2, 3.5),
    "PT": (9.4, 12.5),
    "INR": (0.8, 1.2),
    "NT-proBNP": -1,  # Complex range, often varies significantly especially in heart failure
    "Creatine kinase": (52, 336),  # Assuming range for men
    "Creatinine": (0.74, 1.35),  # Assuming range for men
    "Urea nitrogen": (7, 20),
    "glucose": (70, 100),
    "Blood potassium": (3.5, 5.0),
    "Blood sodium": (135, 145),
    "Blood calcium": (8.5, 10.2),
    "Chloride": (98, 106),
    "Anion gap": (3, 11),
    "Magnesium ion": (1.7, 2.2),
    "PH": (7.35, 7.45),
    "Bicarbonate": (22, 29),
    "Lactic acid": (0.5, 2.2),
    "PCO2": (35, 45),
    "EF": (55, 70)  # Ejection Fraction
    }
    x_original = np.array(x_original)
    counterfactual_p = np.array(counterfactual)
    columns = [f"{i+1}. {feature}" for i, feature in enumerate(features)]
    # Calculate percentage change
    percent_change = calculate_percentage_change(x_original,counterfactual)
    # _,changes_to_normal = adjust_counterfactuals(x_original,counterfactual,features,normal_ranges)
    # Calculate changes needed to reach the normal range
    df = pd.DataFrame({
    'Feature': columns,
    'Original': x_original,
    'Counterfactual': counterfactual,
    'PercentChange': percent_change,
    # 'PercentNormalChange':changes_to_normal
    })
    df.set_index('Feature', inplace=True)
    # plot_feature_changes(df=df)
    significant_changes = df[np.abs(df['PercentChange']) > 0]  # Change the threshold as needed
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    # Plotting the data
    df[['Original', 'Counterfactual']].plot(kind='bar', ax=ax[0])
    ax[0].set_title(f'Comparison of Original Individual(predicted={original_prediction}) and the Counterfactual(predicted={counterfactual_prediction})')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Log-Scaled Value')
    ax[0].set_xlabel('Feature')
    ax[0].tick_params(axis='x', rotation=75) # Rotate feature names for better visibility
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[0].legend(loc='upper right')
    # significant_changes[['PercentChange','PercentNormalChange']].plot(kind='bar', ax=ax[1])
    significant_changes['PercentChange'].plot(kind='bar', ax=ax[1], color='teal')
    ax[1].set_title('Significant Percentage Change of Features')
    ax[1].set_ylabel('Percent Change')
    ax[1].set_xlabel('Feature')
    ax[1].tick_params(axis='x', rotation=75)
    ax[1].grid(True, linestyle='--', linewidth=0.5)
    ax[1].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def generate_all_counterfactuals(X_test, model_predict, generations=250, population_size=100):
    # Array to store all changes
    all_changes = []

    # Loop through each test instance
    for i in range(len(X_test)):
        # Generate counterfactual
        final_population = create_counterfactuals(X_test[i], X_test, 0, model_predict, generations, population_size)
        
        # Assuming the best counterfactual is the first in the final_population
        best_cf = final_population[0]["features"]
        original = X_test[i]
        
        # Calculate changes
        changes = np.abs(best_cf - original) / (np.abs(original) + 1e-6) * 100  # Avoid division by zero
        all_changes.append(changes)

    return np.array(all_changes)    

def save_counterfactual_results(X_test, model_predict, filename):
    all_changes = generate_all_counterfactuals(X_test, model_predict)
    average_changes = np.mean(all_changes, axis=0)
    frequency_of_changes = np.sum(all_changes != 0, axis=0) / len(X_test) * 100

    # Creating DataFrame for better visualization
    features = ["Feature {}".format(i+1) for i in range(all_changes.shape[1])]
    change_df = pd.DataFrame({
        'Feature': features,
        'Average Change (%)': average_changes,
        'Change Frequency (%)': frequency_of_changes
    })
    change_df.to_csv(filename, index=False)

    print(f"Results saved to {filename}")
