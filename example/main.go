package main

import (
	"fmt"

	"github.com/bronystylecrazy/neural"
)

func main() {
	fmt.Println("=== AutoGrad Neural Network ===\n")

	// Create network: 2 inputs -> 8 hidden -> 1 output
	nn := neural.NewNeuralNetwork([]int{2, 8, 1}, "tanh")

	// XOR training data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Training configuration
	config := neural.DefaultTrainConfig()
	config.Epochs = 1000
	config.LR = 0.1
	config.BatchSize = 2 // Mini-batch size
	config.Optimizer = neural.NewAdamW(0.1)
	config.Patience = 100
	config.Verbose = true // Show metrics every epoch

	fmt.Println("Training on XOR problem...")
	metricsHistory := neural.Train(nn, inputs, targets, config)

	// Print final metrics
	if len(metricsHistory) > 0 {
		final := metricsHistory[len(metricsHistory)-1]
		best := metricsHistory[0]
		for _, m := range metricsHistory {
			if m.Accuracy > best.Accuracy {
				best = m
			}
		}
		fmt.Printf("\n=== Final Metrics ===\n")
		fmt.Printf("Loss:      %.6f\n", final.Loss)
		fmt.Printf("Accuracy:  %.2f%%\n", final.Accuracy*100)
		fmt.Printf("Precision: %.6f\n", final.Precision)
		fmt.Printf("Recall:    %.6f\n", final.Recall)
		fmt.Printf("F1 Score:  %.6f\n", final.F1)
		fmt.Printf("\n=== Best Metrics ===\n")
		fmt.Printf("Best Accuracy: %.2f%%\n", best.Accuracy*100)
	}

	// Save model
	err := nn.Save("xor_model.json")
	if err != nil {
		fmt.Println("Failed to save model:", err)
	} else {
		fmt.Println("\nModel saved to xor_model.json")
	}

	fmt.Println("\n=== Predictions ===")
	evaluate(nn, inputs, targets)
}

func evaluate(nn *neural.NeuralNetwork, inputs, targets [][]float64) {
	// Evaluate
	outputs := neural.Evaluate(nn, inputs)
	for i := 0; i < len(inputs); i++ {
		pred := 0.0
		if outputs[i][0] > 0.5 {
			pred = 1.0
		}
		fmt.Printf("  [%v] -> %.4f (predicted: %.0f, target: %.0f)\n",
			inputs[i], outputs[i][0], pred, targets[i][0])
	}
}
