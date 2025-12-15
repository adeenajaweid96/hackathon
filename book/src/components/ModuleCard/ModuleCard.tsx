import React from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import { motion } from 'framer-motion';

interface ModuleCardProps {
  id: number;
  title: string;
  description: string;
  icon: string;
  path: string;
  delay?: number;
}

const ModuleCard: React.FC<ModuleCardProps> = ({
  id,
  title,
  description,
  icon,
  path,
  delay = 0
}) => {
  return (
    <motion.div
      className="col col--3 margin-bottom--lg"
      whileHover={{ y: -10, scale: 1.02 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: delay * 0.1 }}
      style={{ height: '100%' }}
      role="listitem"
    >
      <Link
        to={path}
        className="card card--full-height module-card-link"
        style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
      >
        <div className="card__header text--center padding-horiz--md">
          <div className="margin-bottom--sm">
            <span
              className="text--center module-card-icon"
              style={{
                fontSize: '2.5rem',
                display: 'block',
                margin: '0 auto 1rem'
              }}
            >
              {icon}
            </span>
            <Heading as="h3" className="text--center module-card-title">
              {title}
            </Heading>
          </div>
        </div>
        <div className="card__body text--center">
          <p className="module-card-description">{description}</p>
        </div>
        <div className="card__footer text--center">
          <span
            className="button button--outline button--block module-card-button"
            style={{
              background: 'linear-gradient(90deg, var(--ifm-color-primary), var(--ifm-color-secondary))',
              color: 'white',
              border: 'none',
              fontWeight: 'bold',
              padding: '0.5rem 1rem'
            }}
          >
            Explore Module
          </span>
        </div>
      </Link>
    </motion.div>
  );
};

export default ModuleCard;