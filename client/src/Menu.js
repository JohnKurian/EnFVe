import { Menu, Button } from 'antd';
import React from "react";
import {
    AppstoreOutlined,
    MenuUnfoldOutlined,
    MenuFoldOutlined,
    PieChartOutlined,
    DesktopOutlined,
    ContainerOutlined,
    SettingOutlined,
    LogoutOutlined,
    MailOutlined,
} from '@ant-design/icons';

import { RetweetOutlined, TwitterOutlined } from '@ant-design/icons';


const { SubMenu } = Menu;
class LandingMenu extends React.Component {
    state = {
        collapsed: false,
    };
    toggleCollapsed = () => {
        this.setState({
            collapsed: !this.state.collapsed,
        });
    };
    render() {
        return (
            <div style={{ width: 256 }}>
                <Button type="primary" onClick={this.toggleCollapsed} style={{ marginBottom: 16,"size":"large", 'color':'black', "border": 'None',"backgroundColor":'white' }}>
                    {React.createElement(this.state.collapsed ? MenuUnfoldOutlined : MenuFoldOutlined)}Menu
                </Button>
                <Menu
                    defaultSelectedKeys={['1']}
                    defaultOpenKeys={['sub1']}
                    mode="inline"
                    theme="white"
                    style={{'height':'100vh'}}
                    inlineCollapsed={this.state.collapsed}
                >
                    <Menu.Item key="1" icon={<TwitterOutlined />}>
                        Create Report
                    </Menu.Item>
                    <Menu.Item key="2" icon={<ContainerOutlined />}>
                        Reports
                    </Menu.Item>
                    <SubMenu key="sub1" icon={<AppstoreOutlined />}  title="Social Intelligence Toolkit">
                        <Menu.Item key="5">Fact-Check</Menu.Item>
                        <Menu.Item key="6">Knowledge Query</Menu.Item>
                    </SubMenu>
                    <Menu.Item key="3" icon={<SettingOutlined />}>
                        Settings
                    </Menu.Item>
                    <Menu.Item  key="4" icon={<LogoutOutlined style={{ fontSize: '16px', color: '#f5222d'}} />}>
                        LogOut
                    </Menu.Item>
                </Menu>
            </div>
        );
    }
}
export default LandingMenu;