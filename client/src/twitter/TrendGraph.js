// var trace1 = {
//     x: [1, 2, 3, 4],
//     y: [10, 15, 13, 17],
//     type: 'scatter'
// };
//
// var trace2 = {
//     x: [1, 2, 3, 4],
//     y: [16, 5, 11, 9],
//     type: 'scatter'
// };

// var data = [trace1, trace2];

// Plotly.newPlot('myDiv', data);


import React from 'react';
import Plot from 'react-plotly.js';

class TrendGraph extends React.Component {
    render() {
        return (
            <Plot
                data={[
                    {
                        x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        y: [2, 6, 3, 7, 8, 9, 1, 2, 4, 5, 10],
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {color: 'red'},
                    }
                ]}
                layout={{width: 400, height: 300, title: 'Trend Graph'}}
            />
        );
    }
}

export default TrendGraph;