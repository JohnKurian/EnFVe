import React, { useState, useEffect } from "react";
import Demo from "./Demo";
import {BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend} from "recharts";

class ResultsGraph extends React.Component {
    state = {
            data: [
                {name: 'Page A', uv: 0.2, pv: 2400, amt: 2400},
                {name: 'Page B', uv: 0.01, pv: 1398, amt: 2210},
                {name: 'Page C', uv: 0.1, pv: 9800, amt: 2290}
            ],
            activeIndex: 0,
        }


    handleClick(data, index) {
        this.setState({
            activeIndex: index,
        });
    }

    render () {
        const { activeIndex } = this.state;
        let pred_vals = this.props.evidences.pred_vals
        console.log('pred:', pred_vals)
        let data =[]
        for(let i = 0; i < pred_vals.length; i++) {
            data.push({'uv': pred_vals[i]})
        }
        console.log('data:', data)

        // const activeItem = data[activeIndex];
        const colors = ['#66BB6A', '#FF7043', '#FFEE58']

        return (
            <div>
                <BarChart width={250} height={150} data={data}>
                    <Bar dataKey='uv' onClick={this.handleClick}>
                        {
                            data.map((entry, index) => (
                                <Cell cursor="pointer" fill={colors[index]} key={`cell-${index}`}/>
                            ))
                        }
                    </Bar>
                </BarChart>
            </div>
        );
    }
}


export default ResultsGraph;
