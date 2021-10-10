package linearRegression

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"image/color"
)

func PlotAndSaveData(title string, xAxis string, yAxis string, x *mat.Dense, y *mat.Dense, yp *mat.Dense, m int, path string){
	p := plot.New()

	p.Title.Text = title
	p.X.Label.Text = xAxis
	p.Y.Label.Text = yAxis

	pts := make(plotter.XYs, m)
	for i := 0; i < m; i++ {
		pts[i].X = x.At(0, i)
		pts[i].Y = y.At(0, i)
	}

	s, err := plotter.NewScatter(pts)
	if err != nil {
		panic(err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

	p.Add(s)

	if yp != nil {
		predLine := make(plotter.XYs, m)
		for i := 0; i < m; i++ {
			predLine[i].X = x.At(0, i)
			predLine[i].Y = yp.At(0, i)
		}
		l, err2 := plotter.NewLine(predLine)
		if err2 != nil {
			panic(err2)
		}
		l.LineStyle.Color = color.RGBA{B: 255, A: 255}

		p.Add(l)
	}

	if saveErr := p.Save(4*vg.Inch, 4*vg.Inch, path+".png"); saveErr != nil {
		panic(saveErr)
	}
}
