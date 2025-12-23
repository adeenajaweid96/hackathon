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
      whileHover={{
        y: -15,
        scale: 1.03,
        transition: { duration: 0.2, ease: "easeInOut" }
      }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 30, rotateY: -15 }}
      animate={{ opacity: 1, y: 0, rotateY: 0 }}
      transition={{
        duration: 0.6,
        delay: delay * 0.1,
        ease: [0.25, 0.1, 0.25, 1.0],
        scale: {
          type: "spring",
          stiffness: 300,
          damping: 20
        }
      }}
      style={{ height: '100%' }}
      role="listitem"
    >
      <Link
        to={path}
        className="card card--full-height module-card-link"
        style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
      >
        <div className="card__header text--center padding-horiz--md">
          <motion.div
            className="margin-bottom--sm"
            whileHover={{
              rotate: [0, -5, 5, 0],
              transition: { duration: 0.5 }
            }}
          >
            <motion.span
              className="text--center module-card-icon"
              style={{
                fontSize: '2.5rem',
                display: 'block',
                margin: '0 auto 1rem'
              }}
              whileHover={{
                scale: 1.2,
                transition: { duration: 0.2 }
              }}
            >
              {icon}
            </motion.span>
            <Heading as="h3" className="text--center module-card-title">
              {title}
            </Heading>
          </motion.div>
        </div>
        <div className="card__body text--center">
          <motion.p
            className="module-card-description"
            initial={{ opacity: 0.8 }}
            whileHover={{ opacity: 1 }}
          >
            {description}
          </motion.p>
        </div>
        <div className="card__footer text--center">
          <motion.span
            className="button button--outline button--block module-card-button"
            style={{
              background: 'linear-gradient(90deg, var(--ifm-color-primary), var(--ifm-color-secondary))',
              color: 'white',
              border: 'none',
              fontWeight: 'bold',
              padding: '0.5rem 1rem'
            }}
            whileHover={{
              scale: 1.05,
              boxShadow: '0 8px 25px rgba(0, 0, 0, 0.2)',
              transition: { duration: 0.2 }
            }}
            whileTap={{ scale: 0.95 }}
          >
            Explore Module
          </motion.span>
        </div>
      </Link>
    </motion.div>
  );
};

export default ModuleCard;