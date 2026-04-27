package neural

import (
	"fmt"
	"math"
	"math/rand"

	"go.uber.org/zap"
)

// weightLogger formats weights and biases for logging
type weightLogger struct {
	nn *NeuralNetwork
}

func (w *weightLogger) String() string {
	s := ""
	for li, layer := range w.nn.layers {
		for ni, neuron := range layer.neurons {
			s += fmt.Sprintf("L%dN%d[w:", li, ni)
			for i, weight := range neuron.weights {
				if i > 0 {
					s += ","
				}
				s += fmt.Sprintf("%.4f", weight.data)
			}
			s += fmt.Sprintf(" b:%.4f]", neuron.bias.data)
		}
	}
	return s
}

// TrainConfig holds training configuration
type TrainConfig struct {
	Epochs     int
	LR         float64
	Verbose    bool
	Optimizer  Optimizer
	Scheduler  LRScheduler
	Patience   int     // Early stopping patience (0 = disabled)
	MinDelta   float64 // Minimum change to qualify as improvement
	BatchSize  int     // Mini-batch size (0 = full batch)
	ClipGrad   float64 // Gradient clipping threshold (0 = disabled)
}

// Metrics holds training metrics for a single epoch
type Metrics struct {
	Loss      float64
	Accuracy  float64
	Precision float64
	Recall    float64
	F1        float64
}

// Accuracy computes accuracy given predictions and targets
func Accuracy(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	correct := 0
	for i := range predictions {
		pred := 0.0
		if predictions[i] > 0.5 {
			pred = 1.0
		}
		if pred == targets[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(predictions))
}

// Precision computes precision for binary classification
func Precision(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	truePositives := 0
	predictedPositives := 0
	for i := range predictions {
		pred := 0.0
		if predictions[i] > 0.5 {
			pred = 1.0
		}
		if pred == 1.0 {
			predictedPositives++
			if targets[i] == 1.0 {
				truePositives++
			}
		}
	}
	if predictedPositives == 0 {
		return 0
	}
	return float64(truePositives) / float64(predictedPositives)
}

// Recall computes recall for binary classification
func Recall(predictions []float64, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	truePositives := 0
	actualPositives := 0
	for i := range predictions {
		pred := 0.0
		if predictions[i] > 0.5 {
			pred = 1.0
		}
		if targets[i] == 1.0 {
			actualPositives++
			if pred == 1.0 {
				truePositives++
			}
		}
	}
	if actualPositives == 0 {
		return 0
	}
	return float64(truePositives) / float64(actualPositives)
}

// F1Score computes F1 score from precision and recall
func F1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

// clipGrad clips gradients to the specified threshold
func clipGrad(params []*Value, maxNorm float64) {
	norm := 0.0
	for _, p := range params {
		norm += p.grad * p.grad
	}
	norm = math.Sqrt(norm)

	if norm > maxNorm {
		scale := maxNorm / norm
		for _, p := range params {
			p.grad *= scale
		}
	}
}

// EarlyStop tracks training progress for early stopping
type EarlyStop struct {
	patience  int
	minDelta  float64
	bestLoss  float64
	wait      int
	shouldStop bool
}

// NewEarlyStop creates a new EarlyStop tracker
func NewEarlyStop(patience int, minDelta float64) *EarlyStop {
	return &EarlyStop{
		patience:   patience,
		minDelta:   minDelta,
		bestLoss:   math.Inf(1),
		wait:       0,
		shouldStop: false,
	}
}

// Check evaluates whether to stop training
func (e *EarlyStop) Check(loss float64) bool {
	if loss < e.bestLoss-e.minDelta {
		e.bestLoss = loss
		e.wait = 0
	} else {
		e.wait++
		if e.wait >= e.patience {
			e.shouldStop = true
		}
	}
	return e.shouldStop
}

// DefaultTrainConfig returns default training configuration
func DefaultTrainConfig() *TrainConfig {
	return &TrainConfig{
		Epochs:    1000,
		LR:        0.1,
		Verbose:   true,
		Optimizer:  NewAdamW(0.1),
		Patience:  50,
		MinDelta:  1e-4,
		ClipGrad:  1.0, // Clip gradients to max norm of 1.0
	}
}

// Train trains the neural network on the given data
func Train(nn *NeuralNetwork, inputs, targets [][]float64, config *TrainConfig) []*Metrics {
	params := nn.Parameters()

	// Default optimizer if none provided
	if config.Optimizer == nil {
		config.Optimizer = NewAdamW(config.LR)
	}

	// Default batch size: full batch
	batchSize := config.BatchSize
	if batchSize <= 0 {
		batchSize = len(inputs)
	}

	// Initialize early stopping if enabled
	var earlyStop *EarlyStop
	if config.Patience > 0 {
		earlyStop = NewEarlyStop(config.Patience, config.MinDelta)
	}

	// Create logger once for reuse
	var logger *zap.Logger
	if config.Verbose {
		logger, _ = zap.NewDevelopment()
	}

	// Store metrics history
	metricsHistory := []*Metrics{}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Shuffle indices
		indices := rand.Perm(len(inputs))

		// Collect predictions and targets for metrics
		var epochPredictions []float64
		var epochTargets []float64

		// Process mini-batches
		var epochLoss float64
		numBatches := 0
		for start := 0; start < len(inputs); start += batchSize {
			numBatches++
			end := start + batchSize
			if end > len(inputs) {
				end = len(inputs)
			}
			batchIndices := indices[start:end]

			// Zero gradients at start of batch
			for _, p := range params {
				p.grad = 0.0
			}

			// Forward pass + compute loss for batch
			var batchLoss *Value
			for _, idx := range batchIndices {
				inputVals := make([]*Value, len(inputs[idx]))
				for j := 0; j < len(inputs[idx]); j++ {
					inputVals[j] = NewValue(inputs[idx][j])
				}
				targetVals := make([]*Value, len(targets[idx]))
				for j := 0; j < len(targets[idx]); j++ {
					targetVals[j] = NewValue(targets[idx][j])
				}

				pred := nn.Forward(inputVals)
				sampleLoss := MSE(pred, targetVals)

				// Collect predictions and targets
				epochPredictions = append(epochPredictions, pred[0].data)
				epochTargets = append(epochTargets, targets[idx][0])

				if batchLoss == nil {
					batchLoss = sampleLoss
				} else {
					batchLoss = Add(batchLoss, sampleLoss)
				}
			}

			// Average loss over batch
			batchLoss = Div(batchLoss, NewValue(float64(len(batchIndices))))
			epochLoss += batchLoss.data

			// Backward pass
			batchLoss.Backward()

			// Clip gradients if enabled
			if config.ClipGrad > 0 {
				clipGrad(params, config.ClipGrad)
			}

			// Update parameters
			config.Optimizer.Step(params)
		}

		// Average loss for epoch
		epochLoss /= float64(numBatches)

		// Compute metrics
		accuracy := Accuracy(epochPredictions, epochTargets)
		precision := Precision(epochPredictions, epochTargets)
		recall := Recall(epochPredictions, epochTargets)
		f1 := F1Score(precision, recall)

		metrics := &Metrics{
			Loss:      epochLoss,
			Accuracy:  accuracy,
			Precision: precision,
			Recall:    recall,
			F1:        f1,
		}
		metricsHistory = append(metricsHistory, metrics)

		// Apply learning rate scheduler
		if config.Scheduler != nil {
			config.Scheduler.Step(epoch, config.Optimizer)
		}

		if config.Verbose && logger != nil {
			logger.Info("training progress",
				zap.Int("epoch", epoch),
				zap.Float64("learning_rate", config.Optimizer.GetLR()),
				zap.Float64("loss", epochLoss),
				zap.Float64("accuracy", accuracy),
				zap.Float64("precision", precision),
				zap.Float64("recall", recall),
				zap.Float64("f1", f1),
				zap.Int("batch_size", batchSize),
			)
		}

		// Check early stopping (use loss for early stopping)
		if earlyStop != nil && earlyStop.Check(epochLoss) {
			if logger != nil {
				logger.Info("early stopping triggered",
					zap.Int("final_epoch", epoch),
					zap.Float64("best_loss", earlyStop.bestLoss),
					zap.Float64("final_accuracy", accuracy),
				)
			}
			break
		}
	}

	return metricsHistory
}

// Evaluate evaluates the network and returns predictions
func Evaluate(nn *NeuralNetwork, inputs [][]float64) [][]float64 {
	outputs := make([][]float64, len(inputs))

	for i := 0; i < len(inputs); i++ {
		inputVals := make([]*Value, len(inputs[i]))
		for j := 0; j < len(inputs[i]); j++ {
			inputVals[j] = NewValue(inputs[i][j])
		}
		predVals := nn.Forward(inputVals)

		outputs[i] = make([]float64, len(predVals))
		for j := 0; j < len(predVals); j++ {
			outputs[i][j] = predVals[j].data
		}
	}

	return outputs
}
