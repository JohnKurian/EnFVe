import {Card, Input} from "antd";
import React, { useState, useEffect } from "react";
import { QuestionCircleTwoTone } from '@ant-design/icons';
import {NavLink as RouterNavLink} from "react-router-dom";
import {NavLink} from "reactstrap";

const axios = require('axios')

import {
    BrowserRouter as Router,
    Link,
    Route // for later
} from 'react-router-dom'

const { TextArea } = Input;

function Topic () {
    return (
        <div>
            TOPIC
        </div>
    )
}

class Reports extends React.Component {
    state = {
        value: '',
    };

    componentDidMount() {
        document.title = `You clicked ${this.state.count} times`;
    }

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    render() {
        const { value } = this.state;

        const topics = [{'name': 'report 1', 'id': 1}, {'name': 'report 2', 'id': 2}, {'name': 'report 3', 'id': 3}]

        return (
            <div>
                <ul>
                    {topics.map(({ name, id }) => (
                        <li key={id}>
                            <Link to={`/reports/${id}`}>{name}</Link>
                        </li>
                    ))}
                </ul>


                {/*<Route path={`/reports/:reportId`} component={Topic}/>*/}
            </div>
        )
    }
}

export default Reports;