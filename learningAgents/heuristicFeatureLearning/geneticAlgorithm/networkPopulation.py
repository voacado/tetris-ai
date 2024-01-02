import numpy as np
import torch
import torch.nn as nn

input_size = 6 # number of features
output_size = 1

elitism_pct = 0.2 # keep top 20% of networks
mutation_prob = 0.2 # 20% chance to mutate
weights_init_min = -1
weights_init_max = 1
weights_mutate_power = 0.5

device = 'cpu' # use CPU
# device = torch.device('cuda:0') # use GPU


class Network(nn.Module):
    """
    Represents a network in genetic learning (ex. two parents having children nodes)
    """
    def __init__(self, output_w=None):
        super(Network, self).__init__()
        if not output_w:
            self.output = nn.Linear(
                input_size, output_size, bias=False).to(device)
            self.output.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.output.weight,
                                   a=weights_init_min, b=weights_init_max)
        else:
            self.output = output_w

    def activate(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x = self.output(x)
        return x


class Population:
    """
    Class that represents a genetic being (parent and child).
    """
    def __init__(self, size=25, old_population=None):
        self.size = size
        if old_population is None:
            self.models = [Network() for i in range(size)]
        else:
            # Copy the child
            self.old_models = old_population.models
            self.old_fitnesses = old_population.fitnesses
            self.models = []
            self.crossover()
            self.mutate()
        self.fitnesses = np.zeros(self.size)

    def crossover(self):
        """
        Recombination - combines genetic information (weights) of two parents to form offspring (new children w weights)
        """
        print("Crossover")
        sum_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i] / sum_fitnesses for i in
                 range(self.size)]

        # Sorting descending NNs according to their fitnesses
        sort_indices = np.argsort(probs)[::-1]
        for i in range(self.size):
            if i < self.size * elitism_pct:
                # Add the top performing childs
                model_c = self.old_models[sort_indices[i]]
            else:
                a, b = np.random.choice(self.size, size=2, p=probs,
                                        replace=False)
                prob_neuron_from_a = 0.5

                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = Network()

                for j in range(input_size):
                    # Neuron will come from A with probability
                    # of `prob_neuron_from_a`
                    if np.random.random() > prob_neuron_from_a:
                        model_c.output.weight.data[0][j] = \
                            model_b.output.weight.data[0][j]
                    else:
                        model_c.output.weight.data[0][j] = \
                            model_a.output.weight.data[0][j]

            self.models.append(model_c)

    def mutate(self):
        """
        Explore new data ranges (similar to epsilon-greedy exploration)
        """
        print("Mutating")
        for model in self.models:
            # Mutating weights by adding Gaussian noises
            for i in range(input_size):
                if np.random.random() < mutation_prob:
                    with torch.no_grad():
                        noise = torch.randn(1).mul_(
                            weights_mutate_power).to(device)
                        model.output.weight.data[0][i].add_(noise[0])
