import React, { useState, useEffect } from "react";
import { Card } from 'antd';


class Influencer extends React.Component {
    render() {
        return (
            <Card style={{ width: 300 }}>
                <p>Influencer name</p>
                <p>Tweet content</p>
                <p>Card content</p>
            </Card>
        );
    }
}

export default Influencer;