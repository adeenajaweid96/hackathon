import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import ModuleCard from '@site/src/components/ModuleCard/ModuleCard';
import ChatbotWidget from '@site/src/components/ChatbotWidget/ChatbotWidget';

import styles from './index.module.css';

// Module data for card display
const modules = [
  {
    id: 1,
    title: 'Foundations of Physical AI',
    description: 'Introduction to embodied intelligence, sensor systems, and the fundamentals of physical AI.',
    icon: '🤖',
    path: '/docs/intro-to-physical-ai/intro-to-physical-ai'
  },
  {
    id: 2,
    title: 'ROS 2 - The Robotic Nervous System',
    description: 'Learn about ROS 2 architecture, nodes, topics, services, and robot description formats.',
    icon: '📡',
    path: '/docs/ros-2-fundamentals/ros-2-architecture'
  },
  {
    id: 3,
    title: 'Simulation & Digital Twins',
    description: 'Explore Gazebo simulation, Unity integration, and sensor simulation for humanoid robots.',
    icon: '🎮',
    path: '/docs/gazebo-simulation/setting-up-gazebo'
  },
  {
    id: 4,
    title: 'NVIDIA Isaac Platform',
    description: 'Deep dive into Isaac Sim, synthetic data generation, and perception systems.',
    icon: '👁️',
    path: '/docs/nvidia-isaac/introduction-to-isaac-sim'
  },
  {
    id: 5,
    title: 'Humanoid Robotics Development',
    description: 'Understand kinematics, dynamics, locomotion, and manipulation for humanoid robots.',
    icon: '🦾',
    path: '/docs/humanoid-development/kinematics-dynamics'
  },
  {
    id: 6,
    title: 'Vision-Language-Action Systems',
    description: 'Integrate voice recognition, LLM cognitive planning, and multi-modal interaction.',
    icon: '🧠',
    path: '/docs/vision-language-action/whisper-speech-input'
  },
  {
    id: 7,
    title: 'Capstone: Autonomous Humanoid',
    description: 'Complete project integrating all concepts: path planning, object identification, and manipulation.',
    icon: '🏆',
    path: '/docs/capstone-humanoid-project/system-architecture'
  },
  {
    id: 8,
    title: 'Hardware & Development Infrastructure',
    description: 'Essential hardware requirements, lab setup options, and development infrastructure for humanoid robotics.',
    icon: '🛠️',
    path: '/docs/hardware-requirements/index'
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

function ModuleGrid() {
  return (
    <section className="container padding-vert--lg">
      <div className="row">
        <div className="col col--12">
          <Heading as="h2" className="text--center margin-bottom--xl">
            Course Modules
          </Heading>
          <p className="text--center margin-bottom--lg">
            Explore the comprehensive curriculum designed to take you from foundational concepts to advanced humanoid robotics applications.
          </p>
        </div>
      </div>
      <div className="row">
        {modules.map((module, index) => (
          <ModuleCard
            key={module.id}
            id={module.id}
            title={module.title}
            description={module.description}
            icon={module.icon}
            path={module.path}
            delay={index}
          />
        ))}
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Physical AI and Humanoid Robotics - An AI-native textbook for embodied intelligence">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <ModuleGrid />
      </main>
      {/* Chatbot Widget - Available on all pages */}
      <ChatbotWidget apiUrl="http://localhost:8000/api/v1" />
    </Layout>
  );
}
