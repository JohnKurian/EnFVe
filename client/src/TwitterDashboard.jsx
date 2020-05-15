import {Input} from "antd";
import React, { useState, useEffect } from "react";
import WordCloud  from "./twitter/WordCloud.jsx";
import NewsArticles from './twitter/NewsArticles';
import TweetCard from './twitter/TweetCard';
import Influencer from './twitter/Influencer'

import { Typography } from 'antd';

const { Title } = Typography;

const { TextArea } = Input;



class TwitterDashboard extends React.Component {
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
                <div style={{'display': 'flex', 'flexDirection': 'row'}}>
                    <Title>#coronavirus</Title>
                    <div><WordCloud/></div>
                </div>

                <div style={{'display': 'flex', 'flexDirection': 'row'}}>
                    <div style={{'display': 'flex', 'flexDirection': 'column'}}>
                        <div><TweetCard/></div>
                        <div><TweetCard/></div>
                        <div><TweetCard/></div>
                        <div><TweetCard/></div>
                        <div><TweetCard/></div>
                    </div>
                    <div>
                        <div><Influencer/></div>
                        <div><Influencer/></div>
                        <div><Influencer/></div>
                        <div><Influencer/></div>
                        <div><Influencer/></div>
                    </div>
                </div>


                <div style={{'display': 'flex', 'flexDirection': 'column'}}>
                        <div><NewsArticles/></div>
                </div>

                <div>
                    <div>plotly graph</div>
                </div>


            </div>
    );
    }
}

export default TwitterDashboard;