package neural

import (
	"math"
	"math/rand"
	"time"
)

// ActivationFunction is a function that takes a Value and returns a Value
type ActivationFunction func(*Value) *Value

// Neuron represents a single neuron
type Neuron struct {
	weights   []*Value
	bias      *Value
	actFn     ActivationFunction
	actFnName string // "relu", "sigmoid", "tanh"
}

// NewNeuron creates a neuron with random weights
func NewNeuron(nInputs int, actFn string) *Neuron {
	rand.Seed(time.Now().UnixNano())

	weights := make([]*Value, nInputs)
	for i := 0; i < nInputs; i++ {
		// Xavier initialization
		limit := math.Sqrt(2.0 / float64(nInputs))
		weights[i] = NewValue((rand.Float64()*2 - 1) * limit)
	}
	bias := NewValue(0.0)

	var activationFn ActivationFunction
	switch actFn {
	case "relu":
		activationFn = ReLU
	case "sigmoid":
		activationFn = Sigmoid
	case "tanh":
		activationFn = Tanh
	default:
		activationFn = ReLU
	}

	return &Neuron{
		weights:   weights,
		bias:      bias,
		actFn:     activationFn,
		actFnName: actFn,
	}
}

// Forward computes the neuron's output
func (n *Neuron) Forward(inputs []*Value) *Value {
	sum := n.bias
	for i := 0; i < len(inputs); i++ {
		sum = Add(sum, Mul(inputs[i], n.weights[i]))
	}
	return n.actFn(sum)
}

// Parameters returns all learnable parameters
func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, len(n.weights)+1)
	copy(params, n.weights)
	params[len(n.weights)] = n.bias
	return params
}
