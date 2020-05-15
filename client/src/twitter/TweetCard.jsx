import React, { useState, useEffect } from "react";
import { Card } from 'antd';


class TweetCard extends React.Component {
    render() {
        return (
            <Card style={{ width: 300 }}>
                <p>Tweet name</p>
                <p>Tweet content</p>
                <p>Card content</p>
            </Card>
        );
    }
}

export default TweetCard;