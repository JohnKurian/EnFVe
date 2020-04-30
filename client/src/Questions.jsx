import {Card, Input} from "antd";
import React, { useState, useEffect } from "react";
import { QuestionCircleTwoTone } from '@ant-design/icons';

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
                qa_pairs.map(message =>
                    <Card style={{ width: 375 }}>
                    <div
                        style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-start'}}>
                        <div style={{'fontSize': '22px', 'marginBottom': '12px'}}> <QuestionCircleTwoTone /> {message['question']}</div>
                        <b style={{'fontSize': '23px'}}>{message['answer']}</b>
                    </div>
                    </Card>
                )
            }
            </div>
        )
    }
}

export default Questions;