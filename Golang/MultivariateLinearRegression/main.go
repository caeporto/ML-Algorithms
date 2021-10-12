package main

import (
	"github.com/caeporto/ML-Algorithms/Golang/FeatureNormalization"
	"github.com/caeporto/ML-Algorithms/Golang/GradientDescent"
	"github.com/caeporto/ML-Algorithms/Golang/Util"
	"gonum.org/v1/gonum/mat"
	"strconv"
)

func makeRange(min, max int) []float64 {
	a := make([]float64, max-min+1)
	for i := range a {
		a[i] = float64(min + i)
	}
	return a
}

func main(){
	//2 features
	records := Util.ReadCSV("./resources/data.csv")

	m := len(records)
	features := len(records[0])
	x := mat.NewDense(2, m, nil)
	y := mat.NewDense(1, m, nil)

	for i := range records {
		for feature := 0; feature < features; feature++ {
			if feature == features-1 {
				valy, _ := strconv.ParseFloat(records[i][feature], 64)
				y.Set(0, i, valy)
			} else {
				valx, _ := strconv.ParseFloat(records[i][feature], 64)
				x.Set(feature, i, valx)
			}
		}
	}

	x = FeatureNormalization.NormalizeFeatures(x)

	//adding another row to X after normalization, so we can model the intercept term
	xnorm := mat.NewDense(3, m, nil)

	for i := 0; i < m; i++ {
		xnorm.Set(0, i, 1)
		for j := 1; j < features; j++ {
			xnorm.Set(j, i, x.At(j-1, i))
		}
	}

	theta := mat.NewDense(3, 1, nil)

	iterations := 500
	alpha := 0.005

	costHistory := GradientDescent.GradientDescent(xnorm, y, theta, iterations, alpha, m)
	iter := makeRange(1, iterations)
	xIterations := mat.NewDense(1, iterations, iter)
	yCost := mat.NewDense(1, iterations, costHistory)

	Util.PlotAndSaveData("Cost History", "Iterations", "Cost", xIterations, yCost, 20, nil, iterations, "./resources/cost")

}
