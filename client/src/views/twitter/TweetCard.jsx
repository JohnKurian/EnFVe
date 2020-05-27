import React, { useState, useEffect } from "react";
import {Avatar, Card} from 'antd';
import { RetweetOutlined, TwitterOutlined } from '@ant-design/icons';
import FactCheckPopover from "./FactCheckPopover";


class TweetCard extends React.Component {

    onTweetClick() {
        console.log('on tweet clicked')
    }
    render() {
        return (
            <Card style={{ width: 500, 'margin': '10px', 'boxShadow': '0px 0px 8px -1px rgba(0,0,0,0.33)' }}>
                <p> <Avatar src={this.props.tweet.json.user.profile_image_url} /> {this.props.tweet.json.user.name} <TwitterOutlined onClick={()=> window.open(this.props.tweet.tweet_link, "_blank")}/></p>
                <p>@{this.props.tweet.user_screen_name}</p>
                <p>{this.props.tweet.text}</p>

                <div style={{'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}}>
                    <p><RetweetOutlined/>{this.props.tweet.json.retweet_count}</p>
                    <FactCheckPopover factcheckresults={this.props.factcheckresults}/>
                </div>
            </Card>
        );
    }
}

export default TweetCard;