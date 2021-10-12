package GradientDescent

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

/**
Compute J(theta) using Mean Squared Error
*/
func ComputeCost(x *mat.Dense, y *mat.Dense, theta *mat.Dense, m int) float64{
	//hyp := mat.NewDense(1, m, nil)
	var hyp mat.Dense
	hyp.Mul(theta.T(), x) //linear model, h(x) = theta_0 + theta_1 * x_1 = theta^transposed * x
	hyp.Sub(&hyp, y) // (h(x) - y)
	hyp.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, 2) //(h(x) - y) ^ 2
	}, &hyp)
	cost := (1 / (2 * float64(m))) * mat.Sum(&hyp) // 1/2m * sum((h(x) - y) ^ 2)
	return cost
}

func GradientDescent(x *mat.Dense, y *mat.Dense, theta *mat.Dense, iterations int, alpha float64, m int) []float64{
	//hyp := mat.NewDense(1, m, nil)
	var hyp, gradient mat.Dense
	costHistory := make([]float64, iterations)
	for i := 0; i < iterations; i++ {
		hyp.Mul(theta.T(), x) //linear model, h(x) = theta_0 + theta_1 * x_1 = theta^transposed * x
		hyp.Sub(&hyp, y) // (h(x) - y)
		gradient.Mul(&hyp, x.T())
		gradient.Scale(alpha * 1 / (float64(m)), &gradient)
		theta.Sub(theta, gradient.T())
		costHistory[i] = ComputeCost(x, y, theta, m)
	}

	return costHistory
}
