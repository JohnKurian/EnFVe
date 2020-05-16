import { Layout, Menu } from 'antd';
import React from 'react';
import {
    UserOutlined,
    VideoCameraOutlined,
    UploadOutlined,
} from '@ant-design/icons';

import {
    TwitterOutlined,
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
import TwitterDashboard from "./TwitterDashboard";

const { Header, Sider, Content } = Layout;
const { SubMenu } = Menu;

class BaseLayout extends React.Component {
    state = {
        collapsed: false,
    };

    toggle = () => {
        this.setState({
            collapsed: !this.state.collapsed,
        });
    };

    render() {
        return (
            <Layout>
                <Sider style={{
                    overflow: 'auto',
                    height: '100vh',
                    position: 'fixed',
                    left: 0,
                    "minWidth":"static",
                    // 'width': '250px'
                }} trigger={null} collapsible collapsed={this.state.collapsed}>
                    <div className="logo" />

                    <Menu theme="dark" mode="inline" defaultSelectedKeys={['1']}>
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
                </Sider>



                <Layout className="site-layout">
                    <Header className="site-layout-background" style={{ padding: 0 }}>
                        {React.createElement(this.state.collapsed ? MenuUnfoldOutlined : MenuFoldOutlined, {
                            className: 'trigger',
                            onClick: this.toggle,
                        })}
                    </Header>
                    <Content
                        className="site-layout-background"
                        style={{
                            margin: '24px 16px',
                            padding: 24,
                            minHeight: 280,
                            'marginLeft': '218px'
                        }}
                    >
                        <TwitterDashboard/>
                    </Content>
                </Layout>
            </Layout>
        );
    }
}

export default BaseLayout;