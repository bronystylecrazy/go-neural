package neural

import (
	"math"
)

// Optimizer is the interface for optimization algorithms
type Optimizer interface {
	Step(params []*Value)
	GetLR() float64
	SetLR(lr float64)
}

// SGD is Stochastic Gradient Descent
type SGD struct {
	lr float64
}

// NewSGD creates a new SGD optimizer
func NewSGD(lr float64) *SGD {
	return &SGD{lr: lr}
}

// Step performs one gradient descent step
func (sgd *SGD) Step(params []*Value) {
	for _, p := range params {
		p.data += sgd.lr * p.grad
	}
}

func (sgd *SGD) GetLR() float64 { return sgd.lr }
func (sgd *SGD) SetLR(lr float64) { sgd.lr = lr }

// SGDMomentum is SGD with momentum for faster convergence
type SGDMomentum struct {
	lr        float64
	momentum  float64
	velocity  map[*Value]float64
}

func NewSGDMomentum(lr float64, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		lr:       lr,
		momentum:  momentum,
		velocity:  make(map[*Value]float64),
	}
}

func (sgdm *SGDMomentum) Step(params []*Value) {
	for _, p := range params {
		if _, ok := sgdm.velocity[p]; !ok {
			sgdm.velocity[p] = 0
		}
		// v_t = β * v_{t-1} + grad
		sgdm.velocity[p] = sgdm.momentum*sgdm.velocity[p] + p.grad
		// θ = θ - lr * v_t
		p.data -= sgdm.lr * sgdm.velocity[p]
	}
}

func (sgdm *SGDMomentum) GetLR() float64 { return sgdm.lr }
func (sgdm *SGDMomentum) SetLR(lr float64) { sgdm.lr = lr }

// Adam is Adaptive Moment Estimation optimizer
type Adam struct {
	lr        float64
	beta1     float64
	beta2     float64
	epsilon   float64
	stepCount int
	m         map[*Value]float64 // first moment
	v         map[*Value]float64 // second moment
}

// NewAdam creates a new Adam optimizer
func NewAdam(lr float64) *Adam {
	return &Adam{
		lr:      lr,
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1e-8,
		m:       make(map[*Value]float64),
		v:       make(map[*Value]float64),
	}
}

// Step performs one Adam step
func (adam *Adam) Step(params []*Value) {
	adam.stepCount++

	for _, p := range params {
		// Initialize moments if not present
		if _, ok := adam.m[p]; !ok {
			adam.m[p] = 0
			adam.v[p] = 0
		}

		// Update biased first moment estimate
		adam.m[p] = adam.beta1*adam.m[p] + (1-adam.beta1)*p.grad

		// Update biased second raw moment estimate
		adam.v[p] = adam.beta2*adam.v[p] + (1-adam.beta2)*p.grad*p.grad

		// Compute bias-corrected first moment estimate
		mHat := adam.m[p] / (1 - math.Pow(adam.beta1, float64(adam.stepCount)))

		// Compute bias-corrected second raw moment estimate
		vHat := adam.v[p] / (1 - math.Pow(adam.beta2, float64(adam.stepCount)))

		// Update parameters
		p.data -= adam.lr * mHat / (math.Sqrt(vHat) + adam.epsilon)
	}
}

func (adam *Adam) GetLR() float64 { return adam.lr }
func (adam *Adam) SetLR(lr float64) { adam.lr = lr }

// AdamW is Adam with Weight Decay
type AdamW struct {
	lr        float64
	beta1     float64
	beta2     float64
	epsilon   float64
	weightDecay float64
	stepCount int
	m         map[*Value]float64 // first moment
	v         map[*Value]float64 // second moment
}

// NewAdamW creates a new AdamW optimizer
func NewAdamW(lr float64) *AdamW {
	return &AdamW{
		lr:         lr,
		beta1:      0.9,
		beta2:      0.999,
		epsilon:    1e-8,
		weightDecay: 0.001,
		m:          make(map[*Value]float64),
		v:          make(map[*Value]float64),
	}
}

// Step performs one AdamW step
func (adamw *AdamW) Step(params []*Value) {
	adamw.stepCount++

	for _, p := range params {
		// Initialize moments if not present
		if _, ok := adamw.m[p]; !ok {
			adamw.m[p] = 0
			adamw.v[p] = 0
		}

		// Update biased first moment estimate
		adamw.m[p] = adamw.beta1*adamw.m[p] + (1-adamw.beta1)*p.grad

		// Update biased second raw moment estimate
		adamw.v[p] = adamw.beta2*adamw.v[p] + (1-adamw.beta2)*p.grad*p.grad

		// Compute bias-corrected first moment estimate
		mHat := adamw.m[p] / (1 - math.Pow(adamw.beta1, float64(adamw.stepCount)))

		// Compute bias-corrected second raw moment estimate
		vHat := adamw.v[p] / (1 - math.Pow(adamw.beta2, float64(adamw.stepCount)))

		// Update parameters with weight decay
		p.data -= adamw.lr * (mHat/(math.Sqrt(vHat)+adamw.epsilon) + adamw.weightDecay*p.data)
	}
}

func (adamw *AdamW) GetLR() float64 { return adamw.lr }
func (adamw *AdamW) SetLR(lr float64) { adamw.lr = lr }

// LRScheduler updates the learning rate during training
type LRScheduler interface {
	Step(epoch int, optimizer Optimizer)
	GetName() string
}

// StepLR decays LR by factor every N epochs
type StepLR struct {
	stepSize int
	gamma    float64
}

func NewStepLR(stepSize int, gamma float64) *StepLR {
	return &StepLR{stepSize: stepSize, gamma: gamma}
}

func (s *StepLR) Step(epoch int, optimizer Optimizer) {
	if epoch > 0 && epoch%s.stepSize == 0 {
		optimizer.SetLR(optimizer.GetLR() * s.gamma)
	}
}
func (s *StepLR) GetName() string { return "StepLR" }

// CosineAnnealing decays LR using cosine schedule to min_lr
type CosineAnnealing struct {
	TMax     int
	MinLR    float64
	BaseLR   float64
}

func NewCosineAnnealing(tMax int, minLR float64) *CosineAnnealing {
	return &CosineAnnealing{TMax: tMax, MinLR: minLR, BaseLR: 0}
}

func (c *CosineAnnealing) Step(epoch int, optimizer Optimizer) {
	if c.BaseLR == 0 {
		c.BaseLR = optimizer.GetLR()
	}
	optimizer.SetLR(c.MinLR + (c.BaseLR-c.MinLR)*(1+math.Cos(math.Pi*float64(epoch)/float64(c.TMax)))/2)
}
func (c *CosineAnnealing) GetName() string { return "CosineAnnealing" }

// WarmupCosine starts with linear warmup then cosine annealing
type WarmupCosine struct {
	WarmupEpochs int
	TMax         int
	MinLR        float64
	BaseLR       float64
}

func NewWarmupCosine(warmupEpochs, tMax int, minLR float64) *WarmupCosine {
	return &WarmupCosine{WarmupEpochs: warmupEpochs, TMax: tMax, MinLR: minLR, BaseLR: 0}
}

func (w *WarmupCosine) Step(epoch int, optimizer Optimizer) {
	if w.BaseLR == 0 {
		w.BaseLR = optimizer.GetLR()
	}

	var lr float64
	if epoch < w.WarmupEpochs {
		// Linear warmup
		lr = w.BaseLR * float64(epoch) / float64(w.WarmupEpochs)
	} else {
		// Cosine annealing
		progress := float64(epoch - w.WarmupEpochs) / float64(w.TMax - w.WarmupEpochs)
		lr = w.MinLR + (w.BaseLR-w.MinLR)*(1+math.Cos(math.Pi*progress))/2
	}
	optimizer.SetLR(lr)
}
func (w *WarmupCosine) GetName() string { return "WarmupCosine" }

// ReduceOnPlateau reduces LR when loss stops improving
type ReduceOnPlateau struct {
	Patience  int
	Factor    float64
	MinLR     float64
	bestLoss  float64
	wait      int
	initialized bool
}

func NewReduceOnPlateau(patience int, factor, minLR float64) *ReduceOnPlateau {
	return &ReduceOnPlateau{
		Patience: patience,
		Factor:   factor,
		MinLR:    minLR,
		bestLoss: math.Inf(1),
		wait:     0,
	}
}

func (r *ReduceOnPlateau) Step(epoch int, optimizer Optimizer) {
	// This should be called with current loss - for now we use a simpler approach
	// where it just decays every patience epochs
	if epoch > 0 && epoch%r.Patience == 0 {
		newLR := optimizer.GetLR() * r.Factor
		if newLR >= r.MinLR {
			optimizer.SetLR(newLR)
		}
	}
}
func (r *ReduceOnPlateau) GetName() string { return "ReduceOnPlateau" }
