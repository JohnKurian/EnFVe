import React, { useState, useEffect } from "react";
import { Card } from 'antd';
import { RetweetOutlined, TwitterOutlined } from '@ant-design/icons';
import { Avatar } from 'antd';

class Influencer extends React.Component {

    onTweetClick() {
        console.log('on tweet clicked')
    }
    render() {
        return (
            <Card style={{ width: 300, 'margin': '10px' }}>
                <div style={{'flex': 1, 'flexDirection': 'row'}}>
                <p><Avatar src={this.props.influencer.profile_image_url} /> {this.props.influencer.name} <TwitterOutlined onClick={()=> window.open('https://twitter.com/'+this.props.influencer.screen_name, "_blank")}/></p>
                </div>
                <p style={{'fontSize': '12px'}}>{this.props.influencer.description}</p>
                <p>{this.props.influencer.followers_count} followers</p>

            </Card>
        );
    }
}

export default Influencer;