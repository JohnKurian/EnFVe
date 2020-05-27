import {Card, Input} from "antd";
import React, { useState, useEffect } from "react";
import { QuestionCircleTwoTone } from '@ant-design/icons';
import {NavLink as RouterNavLink} from "react-router-dom";
import {NavLink} from "reactstrap";

import Search from "antd/lib/input/Search";

const { TextArea } = Input;

class CreateReport extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    onSearchClicked(value) {
        console.log('inside', value)
    }

    render() {
        const { value } = this.state;

        return (
            <div>
                <Search
                    placeholder="input search text"
                    onSearch={value => this.onSearchClicked(value)}
                    style={{ width: 200 }}
                />
            </div>
        )
    }
}

export default CreateReport;