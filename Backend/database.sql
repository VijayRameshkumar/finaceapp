CREATE DATABASE IF NOT EXISTS finance_app;

USE finance_app;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    can_access_budget_analysis BOOLEAN DEFAULT FALSE,
    can_access_accounts BOOLEAN DEFAULT FALSE,
    can_access_analysis_report BOOLEAN DEFAULT FALSE,
    can_download BOOLEAN DEFAULT FALSE
);


-- Inser admin user
INSERT INTO users (email, name,password, is_admin, can_access_budget_analysis, can_access_accounts, can_access_analysis_report,can_download) 
VALUES ('admin@gmail.com','Admin', '$2b$12$70pdvIbm1xdnIjP1AzN7nOkqLbXx5Czf3I6g0ZJ1J0OVhjA95vJ8S', TRUE, TRUE, TRUE, TRUE, TRUE);