package neural

import (
	"math/rand"
)

// Layer represents a layer of neurons
type Layer struct {
	neurons []*Neuron
}

// NewLayer creates a layer with nInputs inputs and nNeurons neurons
func NewLayer(nInputs, nNeurons int, actFn string) *Layer {
	neurons := make([]*Neuron, nNeurons)
	for i := 0; i < nNeurons; i++ {
		neurons[i] = NewNeuron(nInputs, actFn)
	}
	return &Layer{neurons: neurons}
}

// Forward computes outputs for all neurons in the layer
func (l *Layer) Forward(inputs []*Value) []*Value {
	outputs := make([]*Value, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Forward(inputs)
	}
	return outputs
}

// Parameters returns all learnable parameters in this layer
func (l *Layer) Parameters() []*Value {
	params := []*Value{}
	for _, neuron := range l.neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

// DropoutLayer applies dropout regularization during training
type DropoutLayer struct {
	rate      float64
	dropped   []bool
	training  bool
}

// NewDropoutLayer creates a new dropout layer
func NewDropoutLayer(rate float64) *DropoutLayer {
	return &DropoutLayer{
		rate:     rate,
		dropped:  nil,
		training: true,
	}
}

// SetTraining sets whether dropout is in training mode
func (d *DropoutLayer) SetTraining(training bool) {
	d.training = training
}

// Forward applies dropout during training
func (d *DropoutLayer) Forward(inputs []*Value) []*Value {
	outputs := make([]*Value, len(inputs))

	if d.training {
		// Generate new dropout mask
		d.dropped = make([]bool, len(inputs))
		for i := range inputs {
			// Keep neuron with probability (1 - rate)
			if rand.Float64() < d.rate {
				d.dropped[i] = true
				outputs[i] = NewValue(0)
			} else {
				d.dropped[i] = false
				// Scale by 1/(1-rate) to maintain expected sum
				outputs[i] = Mul(inputs[i], NewValue(1.0/(1.0-d.rate)))
			}
		}
	} else {
		// During evaluation, no dropout
		for i := range inputs {
			outputs[i] = inputs[i]
		}
	}

	return outputs
}
