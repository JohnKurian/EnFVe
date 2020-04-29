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

        return (
                <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
                    <div style={{'display': 'flex'}}>
                    TOXICITY: {this.props.messages['toxicity_scores']['TOXICITY']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['TOXICITY']*100)}
                    />
                    </div>

                    SEVERE_TOXICITY: {this.props.messages['toxicity_scores']['SEVERE_TOXICITY']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['SEVERE_TOXICITY']*100)}
                    />

                    IDENTITY_ATTACK: {this.props.messages['toxicity_scores']['IDENTITY_ATTACK']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['IDENTITY_ATTACK']*100)}
                    />

                    TOXICITY: {this.props.messages['toxicity_scores']['INSULT']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['INSULT']*100)}
                    />

                    TOXICITY: {this.props.messages['toxicity_scores']['PROFANITY']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['PROFANITY']*100)}
                    />

                    TOXICITY: {this.props.messages['toxicity_scores']['THREAT']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['THREAT']*100)}
                    />

                    TOXICITY: {this.props.messages['toxicity_scores']['SEXUALLY_EXPLICIT']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['SEXUALLY_EXPLICIT']*100)}
                    />


                    TOXICITY: {this.props.messages['toxicity_scores']['FLIRTATION']}
                    <Progress
                        type="circle"
                        strokeColor={{
                            "0%": "#108ee9",
                            "100%": 'red'
                        }}
                        percent={parseInt(this.props.messages['toxicity_scores']['FLIRTATION']*100)}
                    />


                </div>
        );
    }
}

export default Insights;