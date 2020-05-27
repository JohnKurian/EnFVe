import {Card, Input} from "antd";
import React, { useState, useEffect } from "react";
import { QuestionCircleTwoTone } from '@ant-design/icons';
import {NavLink as RouterNavLink} from "react-router-dom";
import {NavLink} from "reactstrap";

const { TextArea } = Input;

class Report extends React.Component {
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
                Report
            </div>
        )
    }
}

export default Report;