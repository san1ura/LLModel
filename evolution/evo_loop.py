"""
Evolution system for transformer models
Self-improvement, mutation, and selection mechanisms
"""
import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
import copy
import os
from dataclasses import dataclass
from tqdm import tqdm
import logging


@dataclass
class EvolutionConfig:
    """
    Configuration for evolution system
    """
    # Mutation parameters
    mutation_rate: float = 0.1
    mutation_strength: float = 0.01
    num_elites: int = 1  # Number of top performers to preserve
    population_size: int = 10  # Number of models in population
    generations: int = 10  # Number of evolutionary generations
    
    # Selection parameters
    tournament_size: int = 3
    selection_pressure: float = 1.5  # Higher means more pressure toward fitter individuals
    
    # Model variation parameters
    weight_perturbation: bool = True  # Whether to perturb weights
    topology_variation: bool = False  # Whether to vary model structure
    layer_droput_rate: float = 0.1  # For topology variation
    
    # Evaluation parameters
    eval_batch_size: int = 4
    eval_num_batches: int = 10


class ModelVariator:
    """
    Class to handle model variations and mutations
    """
    def __init__(self, config: EvolutionConfig):
        self.config = config
    
    def mutate_weights(self, model: nn.Module) -> nn.Module:
        """
        Mutate the weights of a model by adding noise
        """
        mutated_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for param in mutated_model.parameters():
                if random.random() < self.config.mutation_rate:
                    # Add Gaussian noise scaled by mutation strength and parameter magnitude
                    noise = torch.randn_like(param) * self.config.mutation_strength * param.std().item()
                    param.add_(noise)
        
        return mutated_model
    
    def crossover_models(self, model1: nn.Module, model2: nn.Module) -> nn.Module:
        """
        Perform crossover between two models
        """
        child_model = copy.deepcopy(model1)
        
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(), 
                model2.named_parameters()
            ):
                if name1 != name2:
                    raise ValueError(f"Parameter names don't match: {name1} vs {name2}")
                
                # Randomly select weights from either parent (uniform crossover)
                mask = torch.rand_like(param1) > 0.5
                param_child = torch.where(mask, param1, param2)
                child_model.get_parameter(name1).copy_(param_child)
        
        return child_model
    
    def vary_topology(self, model: nn.Module) -> nn.Module:
        """
        Vary the model topology (experimental)
        """
        # This is a simplified approach - in practice, topology variation requires
        # more complex handling to maintain connectivity
        if not self.config.topology_variation:
            return copy.deepcopy(model)
        
        # For now, just return a copy
        return copy.deepcopy(model)


class ScoreEvaluator:
    """
    Evaluate model performance for evolution
    """
    def __init__(self, eval_function: Callable):
        """
        Args:
            eval_function: A function that takes a model and returns a fitness score
        """
        self.eval_function = eval_function
    
    def evaluate_population(self, models: List[nn.Module], 
                           dataloader) -> List[Tuple[nn.Module, float]]:
        """
        Evaluate the fitness of all models in the population
        """
        results = []
        
        for model in tqdm(models, desc="Evaluating population"):
            score = self.eval_function(model, dataloader)
            results.append((model, score))
        
        return results


class EvolutionEngine:
    """
    Main evolution engine that orchestrates the evolutionary process
    """
    def __init__(self, base_model: nn.Module, config: EvolutionConfig, 
                 evaluator: ScoreEvaluator):
        self.base_model = base_model
        self.config = config
        self.evaluator = evaluator
        self.variator = ModelVariator(config)
        
        # Initialize population
        self.population = self.initialize_population()
        
        # Track best model
        self.best_model = copy.deepcopy(base_model)
        self.best_score = float('-inf')
        self.generation_best_scores = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_population(self) -> List[nn.Module]:
        """
        Initialize the population with random variations of the base model
        """
        population = [self.base_model]
        
        for _ in range(self.config.population_size - 1):
            # Create a mutated version of the base model
            mutated_model = self.variator.mutate_weights(self.base_model)
            population.append(mutated_model)
        
        return population
    
    def tournament_selection(self, scored_population: List[Tuple[nn.Module, float]]) -> nn.Module:
        """
        Select a model using tournament selection
        """
        # Select a few random candidates
        tournament_candidates = random.sample(
            scored_population, 
            min(self.config.tournament_size, len(scored_population))
        )
        
        # Sort by score (higher is better) with selection pressure
        # Apply selection pressure by giving preference to higher-scoring members
        tournament_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Probabilistic selection based on rank with selection pressure
        ranks = range(len(tournament_candidates))
        weights = [1.0 / (rank + 1) ** self.config.selection_pressure for rank in ranks]
        
        selected_idx = random.choices(range(len(tournament_candidates)), weights=weights)[0]
        return copy.deepcopy(tournament_candidates[selected_idx][0])
    
    def evolve_generation(self, dataloader) -> List[nn.Module]:
        """
        Evolve the population for one generation
        """
        # Evaluate current population
        scored_population = self.evaluator.evaluate_population(self.population, dataloader)
        
        # Sort by score (best first)
        scored_population.sort(key=lambda x: x[1], reverse=True)
        
        # Update best model if needed
        best_current_model, best_current_score = scored_population[0]
        if best_current_score > self.best_score:
            self.best_model = copy.deepcopy(best_current_model)
            self.best_score = best_current_score
            self.logger.info(f"New best model found with score: {self.best_score:.4f}")
        
        # Track generation best
        self.generation_best_scores.append(best_current_score)
        
        # Keep elites (top performers)
        new_population = [
            copy.deepcopy(model) for model, score in 
            scored_population[:self.config.num_elites]
        ]
        
        # Fill the rest with offspring via selection and mutation
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = self.tournament_selection(scored_population)
            parent2 = self.tournament_selection(scored_population)
            
            # Create offspring through crossover
            if random.random() < 0.5 and parent1 != parent2:
                offspring = self.variator.crossover_models(parent1, parent2)
            else:
                offspring = parent1
            
            # Apply mutation
            offspring = self.variator.mutate_weights(offspring)
            
            new_population.append(offspring)
        
        return new_population
    
    def evolve(self, dataloader, callback: Optional[Callable] = None):
        """
        Run the full evolutionary process
        """
        self.logger.info(f"Starting evolution with {self.config.generations} generations")
        
        for gen in range(self.config.generations):
            self.logger.info(f"Starting generation {gen + 1}/{self.config.generations}")
            
            # Evolve one generation
            self.population = self.evolve_generation(dataloader)
            
            # Report progress
            avg_score = sum(score for _, score in self.evaluator.evaluate_population(self.population, dataloader)) / len(self.population)
            best_gen_score = max(score for _, score in self.evaluator.evaluate_population(self.population, dataloader))
            
            self.logger.info(f"Generation {gen + 1} - Avg Score: {avg_score:.4f}, Best Score: {best_gen_score:.4f}, Global Best: {self.best_score:.4f}")
            
            # Callback if provided
            if callback:
                callback(gen, self.best_model, self.best_score)
        
        self.logger.info(f"Evolution completed. Best score achieved: {self.best_score:.4f}")
        return self.best_model


class SelfImprovementLoop:
    """
    Implements self-improvement mechanisms where the model improves itself
    """
    def __init__(self, model: nn.Module, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Self-improvement components
        self.self_reflection_enabled = True
        self.generate_alternatives_enabled = True
        self.selection_enabled = True
        self.explanation_enabled = True
    
    def generate_reflection(self, input_text: str, generated_output: str) -> str:
        """
        Generate self-reflection on the quality of the model's output
        """
        if not self.self_reflection_enabled:
            return "No reflection needed"
        
        reflection_prompt = (
            f"Input: {input_text}\n\n"
            f"Generated Output: {generated_output}\n\n"
            f"Analyze the quality of the generated output. Identify strengths, weaknesses, and potential improvements:\n"
        )
        
        # Generate reflection using the model itself
        prompt_tokens = self.tokenizer.encode(reflection_prompt, add_special_tokens=True)

        # Validate that all token IDs are within the model's vocabulary size
        vocab_size = self.model.config.vocab_size
        for i, token_id in enumerate(prompt_tokens):
            if token_id >= vocab_size:
                # Replace with unknown token ID if out of bounds
                prompt_tokens[i] = getattr(self.tokenizer, 'unk_token_id', 0)

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

        # Ensure input_ids are within device bounds
        input_ids = input_ids.to(self.model.device)

        reflection = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(reflection[0], skip_special_tokens=True)
    
    def generate_alternative_responses(self, input_text: str, num_alternatives: int = 3) -> List[str]:
        """
        Generate multiple alternative responses for the same input
        """
        if not self.generate_alternatives_enabled:
            return []
        
        alternatives = []
        for _ in range(num_alternatives):
            prompt_tokens = self.tokenizer.encode(input_text, add_special_tokens=True)

            # Validate that all token IDs are within the model's vocabulary size
            vocab_size = self.model.config.vocab_size
            for i, token_id in enumerate(prompt_tokens):
                if token_id >= vocab_size:
                    # Replace with unknown token ID if out of bounds
                    prompt_tokens[i] = getattr(self.tokenizer, 'unk_token_id', 0)

            input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(self.model.device)

            alternative = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
            alternatives.append(self.tokenizer.decode(alternative[0], skip_special_tokens=True))
        
        return alternatives
    
    def select_best_response(self, input_text: str, responses: List[str]) -> Tuple[str, int]:
        """
        Select the best response from a list of alternatives
        """
        if not self.selection_enabled or len(responses) == 0:
            return responses[0] if responses else "", 0
        
        # For simplicity, we'll use a heuristic or another model to score responses
        # In practice, you might use a reward model or human evaluation
        
        # Simple heuristic: prefer longer, more diverse responses
        def score_response(resp):
            # Simple scoring based on length and diversity
            tokens = set(resp.split())
            return len(resp.split()) * 0.5 + len(tokens) * 0.1
        
        scores = [score_response(resp) for resp in responses]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        return responses[best_idx], best_idx
    
    def run_self_improvement_iteration(self, examples: List[Dict[str, str]]) -> nn.Module:
        """
        Run one iteration of self-improvement
        
        Args:
            examples: List of examples with 'input' and 'target' keys
            
        Returns:
            Improved model
        """
        improved_examples = []
        
        for example in tqdm(examples, desc="Self-improvement iteration"):
            input_text = example['input']
            
            # Generate output
            prompt_tokens = self.tokenizer.encode(input_text, add_special_tokens=True)

            # Validate that all token IDs are within the model's vocabulary size
            vocab_size = self.model.config.vocab_size
            for i, token_id in enumerate(prompt_tokens):
                if token_id >= vocab_size:
                    # Replace with unknown token ID if out of bounds
                    prompt_tokens[i] = getattr(self.tokenizer, 'unk_token_id', 0)

            input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(self.model.device)

            generated_output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=150
            )
            generated_text = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
            
            # Generate reflection
            reflection = self.generate_reflection(input_text, generated_text)
            
            # Generate alternatives
            alternatives = self.generate_alternative_responses(input_text, num_alternatives=2)
            
            # Select best among original and alternatives
            all_responses = [generated_text] + alternatives
            best_response, best_idx = self.select_best_response(input_text, all_responses)
            
            improved_examples.append({
                'input': input_text,
                'output': best_response,
                'reflection': reflection
            })
        
        # Now you would typically use these improved examples to fine-tune the model
        # This would involve training on the new examples
        # For now, we'll return the original model
        
        return self.model
    
    def continuous_improvement(self, initial_data: List[Dict[str, str]], 
                              num_iterations: int = 5) -> nn.Module:
        """
        Run continuous self-improvement process
        """
        current_data = initial_data
        current_model = self.model
        
        for iteration in range(num_iterations):
            self.logger.info(f"Self-improvement iteration {iteration + 1}/{num_iterations}")
            
            # Run improvement iteration
            current_model = self.run_self_improvement_iteration(current_data)
            
            # Optionally evaluate the new model and collect more feedback
            # This would involve using the model to generate more data or using external feedback
            pass
        
        return current_model


def create_evolutionary_model_trainer(
    base_model: nn.Module, 
    evaluator: ScoreEvaluator,
    evolution_config: EvolutionConfig,
    training_config: Dict[str, Any]
):
    """
    Create a trainer that combines evolutionary methods with traditional training
    """
    evolution_engine = EvolutionEngine(
        base_model=base_model,
        config=evolution_config,
        evaluator=evaluator
    )
    
    return evolution_engine


def evolution_evaluation_function(model, dataloader):
    """
    Example evaluation function for evolutionary systems
    """
    # This is a placeholder - replace with actual evaluation logic
    # like perplexity, accuracy on a validation set, etc.
    model.eval()

    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= 5:  # Limit evaluation for speed
                break

            inputs = batch['input_ids']
            labels = batch['labels']

            # Validate that input token IDs are within the model's vocabulary size
            vocab_size = model.config.vocab_size
            max_input_id = torch.max(inputs).item()

            if max_input_id >= vocab_size:
                # Log the error but continue with the evaluation
                print(f"Warning: Input token IDs contain values >= vocab_size ({vocab_size}). Max token ID: {max_input_id}")
                # Clamp the values to be within bounds
                inputs = torch.clamp(inputs, max=vocab_size-1)

            outputs = model(inputs)

            # Also validate labels (but respect -100 as ignore index)
            valid_labels = labels != -100  # -100 is the ignore index
            if valid_labels.any():
                max_label_id = torch.max(labels[valid_labels]).item() if valid_labels.any() else -1
                if max_label_id >= vocab_size:
                    print(f"Warning: Label token IDs contain values >= vocab_size ({vocab_size}). Max label ID: {max_label_id}")
                    # Clamp the labels to be within bounds, but preserve the -100 ignore index
                    labels = torch.where(labels == -100, labels, torch.clamp(labels, max=vocab_size-1))

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            count += 1

    # Convert loss to fitness score (higher is better)
    avg_loss = total_loss / count if count > 0 else float('inf')
    fitness_score = -avg_loss  # Negative because lower loss is better

    return fitness_score