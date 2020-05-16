import React, { useState, useEffect } from "react";
import {Avatar, Card} from 'antd';
import { RetweetOutlined, TwitterOutlined } from '@ant-design/icons';


class TweetCard extends React.Component {

    onTweetClick() {
        console.log('on tweet clicked')
    }
    render() {
        return (
            <Card style={{ width: 300 }}>
                <p> <Avatar src={this.props.tweet.json.user.profile_image_url} /> {this.props.tweet.json.user.name} <TwitterOutlined onClick={()=> window.open(this.props.tweet.tweet_link, "_blank")}/></p>
                <p>@{this.props.tweet.user_screen_name}</p>
                <p>{this.props.tweet.text}</p>
                <p><RetweetOutlined />{this.props.tweet.json.retweet_count}</p>
            </Card>
        );
    }
}

export default TweetCard;