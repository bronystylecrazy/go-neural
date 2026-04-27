package neural

// MSE computes Mean Squared Error
func MSE(predictions, targets []*Value) *Value {
	n := float64(len(predictions))
	sum := NewValue(0.0)
	for i := 0; i < len(predictions); i++ {
		diff := Sub(predictions[i], targets[i])
		sum = Add(sum, Pow(diff, 2))
	}
	return Div(sum, NewValue(n))
}

// BinaryCrossEntropy for binary classification
func BinaryCrossEntropy(predictions, targets []*Value) *Value {
	eps := 1e-8
	sum := NewValue(0.0)
	for i := 0; i < len(predictions); i++ {
		// -(target * log(pred) + (1-target) * log(1-pred))
		predClamped := Add(predictions[i], NewValue(eps))
		oneMinusPred := Sub(NewValue(1.0), predictions[i])
		oneMinusTarget := Sub(NewValue(1.0), targets[i])

		term1 := Mul(targets[i], Log(predClamped))
		term2 := Mul(oneMinusTarget, Log(Add(oneMinusPred, NewValue(eps))))
		sum = Add(sum, Mul(NewValue(-1.0), Add(term1, term2)))
	}
	return Div(sum, NewValue(float64(len(predictions))))
}
