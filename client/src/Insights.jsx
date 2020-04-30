import {Input, Progress} from "antd";
import React, { useState, useEffect } from "react";
const { TextArea } = Input;

class Insights extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    render() {
        const { value } = this.state;

        const stateMap = {
            'TOXICITY': 'Toxicity',
            'SEVERE_TOXICITY': 'Severe Toxicity',
            'IDENTITY_ATTACK': 'Identity Attack',
            'INSULT': 'Insult',
            'PROFANITY': 'Profanity',
            'THREAT': 'Threat',
            'SEXUALLY_EXPLICIT': 'Sexually Explicit',
            'FLIRTATION': 'Flirtation'
        }

        return (
                <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-end', 'marginTop': '50px'}}>

                    { Object.keys(stateMap).map( key =>
                        <div style={{'display': 'flex', 'alignItems': 'center'}}>
                            <div style={{'font-size': '25px', 'margin': '25px'}}>
                                {stateMap[key]}
                            </div>
                            <div style={{'display': 'flex'}}>
                                <Progress
                                    type="circle"
                                    strokeColor={{
                                        "0%": "#108ee9",
                                        "100%": 'red'
                                    }}
                                    width='60px'
                                    height='60px'
                                    percent={parseInt(this.props.messages['toxicity_scores'][key] * 100)}
                                />
                            </div>
                        </div>
                    )
                    }
                </div>

        );
    }
}

export default Insights;