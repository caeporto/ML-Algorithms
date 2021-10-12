package NormalEquation

import (
	"gonum.org/v1/gonum/mat"
)

func NormalEquation(x *mat.Dense, y *mat.Dense) *mat.Dense{

	var theta, left, right mat.Dense
	left.Mul(x.T(), x)
	left.Inverse(&left)
	right.Mul(x.T(), y)
	theta.Mul(&left, &right)

	return &theta
}
