import React from "react";
import "../App.css";
import { Avatar } from 'antd';
// import { Icon, InlineIcon } from '@iconify/react';
// import betaIcon from '@iconify/icons-mdi/beta';
// import {UserOutlined} from '@ant-design/icons';
function Topheader(){
    return(
        <div className="card " style={{
            'height': '100px',
            'position': 'sticky',
            // 'background-image': 'linear-gradient(to bottom right, black ,white)',
            'backgroundColor':'white',
            'top': '0',
            'zIndex': '26'/* this is optional and should be different for every project */,
        }}>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'justifyContent': 'space-between','color':'black','font-size':'20px'}}>
                {/*<div>*/}
                {/*    /!*<img src={ require('./logo.png') } />*!/*/}
                {/*    Nexus Detective AI<Icon style= {{"margin-bottom": "-20px","margin-right":"50px" }}  icon={betaIcon} /></div>*/}
                <div style={{'display': 'flex','flexDirection': 'row','alignItems': 'flex-end', "marginRight":"15px"}}>
                    <Avatar src="https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png" />
                    <button style={{'font-size':'20px', 'color':'black', 'background': 'none','border':'none'}}>Login</button>
                </div>
            </div>
        </div>
    );
}
export default Topheader;