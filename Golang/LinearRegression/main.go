package main

import (
	"github.com/caeporto/ML-Algorithms/Golang/GradientDescent"
	"github.com/caeporto/ML-Algorithms/Golang/Util"
	"gonum.org/v1/gonum/mat"
	"strconv"
)

func main(){
	resources := Util.ReadCSV("./resources/data.csv")

	m := len(resources)
	x := mat.NewDense(2, m, nil) //adding another row to X, so we can model the intercept term
	y := mat.NewDense(1, m, nil)

	for i := range resources {
		x.Set(0, i, 1)
		valx, _ := strconv.ParseFloat(resources[i][0], 64)
		x.Set(1, i, valx)
		valy, _ := strconv.ParseFloat(resources[i][1], 64)
		y.Set(0, i, valy)
	}

	X := x.Slice(1, 2, 0, m).(*mat.Dense) //get just the X-Axis without the additional row

	Util.PlotAndSaveData("Food Truck", "Population (in 10.000s)", "Profit (in $10.000s)", X, y, 1,nil, m, "./resources/points")

	theta := mat.NewDense(2, 1, nil) //theta_0 and theta_1 terms

	iterations := 1500
	alpha := 0.01

	costHistory := GradientDescent.GradientDescent(x, y, theta, iterations, alpha, m)
	iter := Util.MakeRange(1, iterations)
	xIterations := mat.NewDense(1, iterations, iter)
	yCost := mat.NewDense(1, iterations, costHistory)

	var ypred mat.Dense
	ypred.Mul(theta.T(), x)

	Util.PlotAndSaveData("Food Truck", "Population (in 10.000s)", "Profit (in $10.000s)", X, y, 1, &ypred, m, "./resources/predicted")

	Util.PlotAndSaveData("Cost History", "Iterations", "Cost", xIterations, yCost, 10, nil, iterations, "./resources/cost")

}
