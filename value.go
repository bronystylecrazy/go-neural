package neural

import (
	"fmt"
	"math"
)

// Value represents a node in the computation graph
type Value struct {
	data       float64
	grad       float64
	op         string
	prev       []*Value
	backwardFn func()
}

// NewValue creates a leaf value (no operation)
func NewValue(data float64) *Value {
	return &Value{
		data: data,
		grad: 0,
		op:   "",
		prev: nil,
	}
}

// Data returns the value's data
func (v *Value) Data() float64 { return v.data }

// Grad returns the value's gradient
func (v *Value) Grad() float64 { return v.grad }

// Add: a + b
func Add(a, b *Value) *Value {
	out := &Value{
		data: a.data + b.data,
		op:   "+",
		prev: []*Value{a, b},
	}

	out.backwardFn = func() {
		a.grad += out.grad * 1.0
		b.grad += out.grad * 1.0
	}

	return out
}

// Sub: a - b
func Sub(a, b *Value) *Value {
	return Add(a, Mul(b, NewValue(-1)))
}

// Mul: a * b
func Mul(a, b *Value) *Value {
	out := &Value{
		data: a.data * b.data,
		op:   "*",
		prev: []*Value{a, b},
	}

	out.backwardFn = func() {
		a.grad += out.grad * b.data
		b.grad += out.grad * a.data
	}

	return out
}

// Div: a / b
func Div(a, b *Value) *Value {
	out := &Value{
		data: a.data / b.data,
		op:   "/",
		prev: []*Value{a, b},
	}

	out.backwardFn = func() {
		a.grad += out.grad * (1.0 / b.data)
		b.grad += out.grad * (-a.data / (b.data * b.data))
	}

	return out
}

// Pow: a^exp
func Pow(a *Value, exp float64) *Value {
	out := &Value{
		data: math.Pow(a.data, exp),
		op:   fmt.Sprintf("pow%.0f", exp),
		prev: []*Value{a},
	}

	out.backwardFn = func() {
		a.grad += out.grad * exp * math.Pow(a.data, exp-1)
	}

	return out
}

// Neg: -a
func Neg(a *Value) *Value {
	return Mul(NewValue(-1), a)
}

// ReLU: max(0, x)
func ReLU(x *Value) *Value {
	out := &Value{
		data: math.Max(0, x.data),
		op:   "ReLU",
		prev: []*Value{x},
	}

	out.backwardFn = func() {
		if x.data > 0 {
			x.grad += out.grad * 1.0
		}
	}

	return out
}

// Sigmoid: 1 / (1 + e^(-x))
func Sigmoid(x *Value) *Value {
	out := &Value{
		data: 1.0 / (1.0 + math.Exp(-x.data)),
		op:   "sigmoid",
		prev: []*Value{x},
	}

	out.backwardFn = func() {
		x.grad += out.grad * out.data * (1.0 - out.data)
	}

	return out
}

// Tanh: hyperbolic tangent
func Tanh(x *Value) *Value {
	out := &Value{
		data: math.Tanh(x.data),
		op:   "tanh",
		prev: []*Value{x},
	}

	out.backwardFn = func() {
		x.grad += out.grad * (1.0 - out.data*out.data)
	}

	return out
}

// Log: natural logarithm
func Log(x *Value) *Value {
	out := &Value{
		data: math.Log(x.data),
		op:   "log",
		prev: []*Value{x},
	}

	out.backwardFn = func() {
		x.grad += out.grad * (1.0 / x.data)
	}

	return out
}

// Exp: e^x
func Exp(x *Value) *Value {
	out := &Value{
		data: math.Exp(x.data),
		op:   "exp",
		prev: []*Value{x},
	}

	out.backwardFn = func() {
		x.grad += out.grad * out.data
	}

	return out
}

// Backward computes gradients by traversing the computation graph
func (v *Value) Backward() {
	// Build topological order
	order := []*Value{}
	visited := make(map[*Value]bool)

	var buildOrder func(*Value)
	buildOrder = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.prev {
			buildOrder(child)
		}
		order = append(order, v)
	}
	buildOrder(v)

	// Process in reverse topological order
	v.grad = 1.0
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backwardFn != nil {
			order[i].backwardFn()
		}
	}
}

// ZeroGrad sets gradient to zero
func (v *Value) ZeroGrad() {
	v.grad = 0.0
}

// Softmax computes softmax for a slice of Values
// softmax(x_i) = exp(x_i) / sum(exp(x_j))
func Softmax(logits []*Value) []*Value {
	// Find max for numerical stability
	maxVal := logits[0].data
	for _, l := range logits {
		if l.data > maxVal {
			maxVal = l.data
		}
	}

	// Compute exp and sum
	expVals := make([]*Value, len(logits))
	sumExp := NewValue(0)
	for i, l := range logits {
		expVals[i] = Exp(Sub(l, NewValue(maxVal)))
		sumExp = Add(sumExp, expVals[i])
	}

	// Normalize
	output := make([]*Value, len(logits))
	for i, e := range expVals {
		output[i] = Div(e, sumExp)
	}

	return output
}

// CrossEntropy computes cross-entropy loss: -sum(y * log(p))
func CrossEntropy(logits []*Value, target int) *Value {
	// Apply softmax
	probs := Softmax(logits)

	// Cross entropy: -log(probs[target])
	// Add small epsilon for numerical stability
	loss := Neg(Log(Add(probs[target], NewValue(1e-15))))

	return loss
}

// CrossEntropyBatch computes cross-entropy loss for a batch
func CrossEntropyBatch(logits []*Value, targets []int) *Value {
	batchSize := len(targets)
	if batchSize == 0 {
		return NewValue(0)
	}

	loss := NewValue(0)
	for i, target := range targets {
		// Get logits for sample i (assuming logits are grouped by sample)
		sampleLogits := logits[i*batchSize : (i+1)*batchSize]
		sampleLoss := CrossEntropy(sampleLogits, target)
		loss = Add(loss, sampleLoss)
	}

	return Div(loss, NewValue(float64(batchSize)))
}
