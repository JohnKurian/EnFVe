import {Card, Input} from "antd";
import React, { useState, useEffect } from "react";
import { QuestionCircleTwoTone } from '@ant-design/icons';
import {NavLink as RouterNavLink} from "react-router-dom";
import {NavLink} from "reactstrap";

import Search from "antd/lib/input/Search";


import { Button } from "reactstrap";
import Highlight from "../components/Highlight";
import { useAuth0 } from "../react-auth0-spa";
import config from "../auth_config.json";
import TwitterDashboard from "./TwitterDashboard";



const { apiOrigin = "http://127.0.0.1:3001" } = config;

const CreateReport = () => {
    const [showResult, setShowResult] = useState(false);
    const [apiMessage, setApiMessage] = useState("");
    const { getTokenSilently, user } = useAuth0();

    console.log('user:', user)

    const callApi = async (value) => {
        try {
            const token = await getTokenSilently();

            const response = await fetch(`${apiOrigin}/api/createreport`, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    'hashtags': [value],
                    'user': user
                })
            });

            const responseData = await response.json();

            setShowResult(true);
            setApiMessage(responseData);
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <>
            <div className="mb-5">
                <h1>External API</h1>
                <p>
                    Ping an external API by clicking the button below. This will call the
                    external API using an access token, and the API will validate it using
                    the API's audience value.
                </p>

                <Button color="primary" className="mt-5" onClick={callApi}>
                    Ping API
                </Button>

                <Search
                    placeholder="input search text"
                    onSearch={value => callApi(value)}
                    style={{ width: 200 }}
                />

            </div>

            <div className="result-block-container">
                <div className={`result-block ${showResult && "show"}`}>
                    <h6 className="muted">Result</h6>
                    <Highlight>{JSON.stringify(apiMessage, null, 2)}</Highlight>
                </div>
            </div>

        </>
    );
};

export default CreateReport;
