import {Input} from "antd";
import React, { useState, useEffect } from "react";
const { TextArea } = Input;

class Questions extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    render() {
        const { value } = this.state;
        const qa_pairs = this.props.messages['qa_pairs']

        return (
            <div>
            {
                qa_pairs.map(message => <div
                        style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}}>
                        <div>{message['question']}</div>
                        <div>{message['answer']}</div>
                    </div>
                )
            }
            </div>
        )
    }
}

export default Questions;