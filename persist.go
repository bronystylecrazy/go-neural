package neural

import (
	"encoding/json"
	"os"
)

// ModelState represents the serializable state of a neural network
type ModelState struct {
	LayerSizes []int    `json:"layer_sizes"`
	Activation string   `json:"activation"`
	Weights    [][][]float64 `json:"weights"` // [layer][neuron][weight]
	Biases     [][]float64   `json:"biases"`  // [layer][neuron]
}

// Save saves the neural network weights to a JSON file
func (nn *NeuralNetwork) Save(filepath string) error {
	state := ModelState{
		LayerSizes: make([]int, len(nn.layers)+1),
		Activation: "",
		Weights:    make([][][]float64, len(nn.layers)),
		Biases:     make([][]float64, len(nn.layers)),
	}

	// Get layer sizes from first layer
	state.LayerSizes[0] = len(nn.layers[0].neurons[0].weights)
	for i, layer := range nn.layers {
		if i == 0 {
			state.LayerSizes[i] = len(layer.neurons[0].weights)
		}
		state.LayerSizes[i+1] = len(layer.neurons)

		state.Weights[i] = make([][]float64, len(layer.neurons))
		state.Biases[i] = make([]float64, len(layer.neurons))

		for j, neuron := range layer.neurons {
			state.Weights[i][j] = make([]float64, len(neuron.weights))
			for k, w := range neuron.weights {
				state.Weights[i][j][k] = w.data
			}
			state.Biases[i][j] = neuron.bias.data
		}
	}

	// Try to determine activation function from first neuron
	if len(nn.layers) > 0 && len(nn.layers[0].neurons) > 0 {
		state.Activation = nn.layers[0].neurons[0].actFnName
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath, data, 0644)
}

// Load loads the neural network weights from a JSON file
func (nn *NeuralNetwork) Load(filepath string) error {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return err
	}

	var state ModelState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}

	// Verify layer sizes match
	expectedSizes := make([]int, len(nn.layers)+1)
	if len(nn.layers) > 0 && len(nn.layers[0].neurons) > 0 {
		expectedSizes[0] = len(nn.layers[0].neurons[0].weights)
		for i := range nn.layers {
			expectedSizes[i+1] = len(nn.layers[i].neurons)
		}
	}

	if len(expectedSizes) != len(state.LayerSizes) {
		return ErrLayerSizeMismatch
	}
	for i, size := range expectedSizes {
		if i > 0 && size != state.LayerSizes[i] {
			return ErrLayerSizeMismatch
		}
	}

	// Load weights and biases
	for i, layer := range nn.layers {
		for j, neuron := range layer.neurons {
			for k := range neuron.weights {
				neuron.weights[k].data = state.Weights[i][j][k]
			}
			neuron.bias.data = state.Biases[i][j]
		}
	}

	return nil
}

// NewNeuralNetworkFromFile creates a new network and loads weights from a file
func NewNeuralNetworkFromFile(filepath string) (*NeuralNetwork, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	var state ModelState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}

	nn := NewNeuralNetwork(state.LayerSizes, state.Activation)
	if err := nn.Load(filepath); err != nil {
		return nil, err
	}

	return nn, nil
}

// Custom errors
var ErrLayerSizeMismatch = &LayerSizeMismatchError{}

type LayerSizeMismatchError struct{}

func (e *LayerSizeMismatchError) Error() string {
	return "layer size mismatch: cannot load weights into network with different architecture"
}
