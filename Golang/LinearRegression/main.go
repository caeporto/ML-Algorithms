package main

import (
	"LinearRegression/linearRegression"
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
	resources := linearRegression.ReadCSV("./resources/data.csv")

	m := len(resources)
	x := mat.NewDense(2, m, nil) //adding another row to X, so we can model the intercept term
	y := mat.NewDense(1, m, nil)

	for i, _ := range resources {
		x.Set(0, i, 1)
		valx, _ := strconv.ParseFloat(resources[i][0], 64)
		x.Set(1, i, valx)
		valy, _ := strconv.ParseFloat(resources[i][1], 64)
		y.Set(0, i, valy)
	}

	X := x.Slice(1, 2, 0, m).(*mat.Dense) //get just the X-Axis without the additional row

	linearRegression.PlotAndSaveData("Food Truck", "Population (in 10.000s)", "Profit (in $10.000s)", X, y, 1,nil, m, "./resources/points")

	theta := mat.NewDense(2, 1, nil) //theta_0 and theta_1 terms

	iterations := 1500
	alpha := 0.01

	costHistory := linearRegression.GradientDescent(x, y, theta, iterations, alpha, m)
	iter := makeRange(1, iterations)
	xIterations := mat.NewDense(1, iterations, iter)
	yCost := mat.NewDense(1, iterations, costHistory)

	var ypred mat.Dense
	ypred.Mul(theta.T(), x)

	linearRegression.PlotAndSaveData("Food Truck", "Population (in 10.000s)", "Profit (in $10.000s)", X, y, 1, &ypred, m, "./resources/predicted")

	linearRegression.PlotAndSaveData("Cost History", "Iterations", "Cost", xIterations, yCost, 10, nil, iterations, "./resources/cost")

}
