import {Card, Input} from "antd";
import React, { useState, useEffect } from "react";
import { QuestionCircleTwoTone } from '@ant-design/icons';
import {NavLink as RouterNavLink} from "react-router-dom";
import {NavLink} from "reactstrap";
import { useAuth0 } from "../react-auth0-spa";



import {
    BrowserRouter as Router,
    Link,
    Route // for later
} from 'react-router-dom'
import config from "../auth_config.json";


const { apiOrigin = "http://localhost:3001" } = config;

const axios = require('axios')

const { TextArea } = Input;

function Topic () {
    return (
        <div>
            TOPIC
        </div>
    )
}

const Reports = () => {
    const { getTokenSilently, user } = useAuth0();

    const [reports, setReports] = useState([]);


    const callApi = async () => {
        try {
            const token = await getTokenSilently();

            const response = await fetch(`${apiOrigin}/api/get_reports`, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    'user_profile': user
                })
            });

            const responseData = await response.json();
            setReports(responseData['reports']);
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        callApi()
    }, []);


    const onChange = ({ target: { value } }) => {
        this.setState({ value });
    };


        const topics = [{'name': 'report 1', 'id': 1}, {'name': 'report 2', 'id': 2}, {'name': 'report 3', 'id': 3}]


        return (
            <div>
                <ul>
                    {reports && reports.map(({ hashtags, id }) => (
                        <li key={id}>
                            <Link to={`/reports/${id}`}>{id}</Link>
                        </li>
                    ))}
                </ul>


                {/*<Route path={`/reports/:reportId`} component={Topic}/>*/}
            </div>
        )

}

export default Reports;