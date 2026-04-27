package neural

// NeuralNetwork represents a feedforward neural network
type NeuralNetwork struct {
	layers     []*Layer
	dropouts   []*DropoutLayer
	training   bool
}

// NewNeuralNetwork creates a network with given architecture
// Example: []int{2, 4, 3, 1} -> 2 inputs, 4 hidden, 3 hidden, 1 output
func NewNeuralNetwork(layerSizes []int, actFn string) *NeuralNetwork {
	layers := make([]*Layer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		layers[i] = NewLayer(layerSizes[i], layerSizes[i+1], actFn)
	}
	return &NeuralNetwork{
		layers:   layers,
		dropouts: nil,
		training: true,
	}
}

// NewNeuralNetworkWithDropout creates a network with dropout layers
// dropouts parameter specifies dropout rate for each hidden layer (0 = no dropout)
func NewNeuralNetworkWithDropout(layerSizes []int, actFn string, dropouts []float64) *NeuralNetwork {
	layers := make([]*Layer, len(layerSizes)-1)
	dropoutLayers := make([]*DropoutLayer, len(layerSizes)-2) // Dropout between layers

	for i := 0; i < len(layerSizes)-1; i++ {
		layers[i] = NewLayer(layerSizes[i], layerSizes[i+1], actFn)
		// Add dropout after this layer if specified
		if i < len(dropouts) && dropouts[i] > 0 {
			dropoutLayers[i] = NewDropoutLayer(dropouts[i])
		}
	}

	return &NeuralNetwork{
		layers:   layers,
		dropouts: dropoutLayers,
		training: true,
	}
}

// SetTraining sets training mode (affects dropout)
func (nn *NeuralNetwork) SetTraining(training bool) {
	nn.training = training
	for _, d := range nn.dropouts {
		if d != nil {
			d.SetTraining(training)
		}
	}
}

// Forward computes the network output
func (nn *NeuralNetwork) Forward(inputs []*Value) []*Value {
	current := inputs
	for i, layer := range nn.layers {
		current = layer.Forward(current)
		// Apply dropout after this layer (if any)
		if i < len(nn.dropouts) && nn.dropouts[i] != nil {
			current = nn.dropouts[i].Forward(current)
		}
	}
	return current
}

// Parameters returns all learnable parameters
func (nn *NeuralNetwork) Parameters() []*Value {
	params := []*Value{}
	for _, layer := range nn.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
