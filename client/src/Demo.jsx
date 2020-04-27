import {Input} from "antd";
import React, { useState, useEffect } from "react";
const { TextArea } = Input;

class Demo extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    render() {
        const { value } = this.state;

        return (
            <div>
                <TextArea placeholder="Autosize height based on content lines" autoSize style={{'height': '32px'}}/>
                <div style={{ margin: '24px 0' }} />
            </div>
        );
    }
}

export default Demo;