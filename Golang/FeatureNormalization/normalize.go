package FeatureNormalization

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

func stdDeviationFeatures(XFeat mat.Dense) *mat.Dense{
	features, _ := XFeat.Dims()
	sigma := make([]float64, features)
	for feature := 0; feature < features; feature++ {
		featureRow := XFeat.RawRowView(feature)
		stdDev := math.Sqrt(stat.Variance(featureRow, nil))
		sigma[feature] = stdDev
	}
	return mat.NewDense(1, features, sigma)
}

func meanFeatures(XFeat mat.Dense) *mat.Dense{
	features, _ := XFeat.Dims()
	mu := make([]float64, features)
	for feature := 0; feature < features; feature++ {
		featureRow := XFeat.RawRowView(feature)
		mean := stat.Mean(featureRow, nil)
		mu[feature] = mean
	}
	return mat.NewDense(1, features, mu)
}

func NormalizeFeatures(X *mat.Dense) *mat.Dense{
	var XNorm mat.Dense
	XNorm.CloneFrom(X)
	features, values := XNorm.Dims()

	mu := meanFeatures(XNorm)
	sigma := stdDeviationFeatures(XNorm)

	for feature := 0; feature < features; feature++ {
		for value := 0; value < values; value++ {
			XNorm.Set(feature, value, XNorm.At(feature, value) - mu.At(0, feature))
			XNorm.Set(feature, value, XNorm.At(feature, value) / sigma.At(0, feature))
		}
	}

	return &XNorm
}
