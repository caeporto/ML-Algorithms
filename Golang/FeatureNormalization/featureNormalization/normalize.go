package featureNormalization

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func stdDeviationFeatures(XFeat mat.Dense, mean mat.Dense) *mat.Dense{
	features, values := XFeat.Dims()
	sigma := make([]float64, features)
	for feature := 0; feature < features; feature++ {
		featSum := 0.0
		sigma[feature] = 0
		for value := 0; value < values; value++ {
			featSum += math.Pow(math.Abs(XFeat.At(feature, value) - mean.At(0, feature)), 2)
		}
		featSum /= float64(values)
		sigma[feature] = math.Sqrt(featSum)
	}
	return mat.NewDense(1, features, sigma)
}

func meanFeatures(XFeat mat.Dense) *mat.Dense{
	features, values := XFeat.Dims()
	mu := make([]float64, features)
	for feature := 0; feature < features; feature++ {
		mu[feature] = 0
		for value := 0; value < values; value++ {
			mu[feature] += XFeat.At(feature, value)
		}
		mu[feature] /= float64(values)
	}
	return mat.NewDense(1, features, mu)
}

func NormalizeFeatures(X *mat.Dense) *mat.Dense{
	var XNorm mat.Dense
	XNorm.CloneFrom(X)
	features, values := XNorm.Dims()

	mu := meanFeatures(XNorm)
	sigma := stdDeviationFeatures(XNorm, *mu)

	for feature := 0; feature < features; feature++ {
		for value := 0; value < values; value++ {
			XNorm.Set(feature, value, XNorm.At(feature, value) - mu.At(0, feature))
			XNorm.Set(feature, value, XNorm.At(feature, value) / sigma.At(0, feature))
		}
	}

	return &XNorm
}
