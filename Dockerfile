# Use an official Node.js runtime as a parent image
FROM node:20

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install any needed packages
RUN npm install

# Bundle app source
COPY . .

# Compile TypeScript to JavaScript
RUN npm install -g typescript
RUN tsc

# Creates a non-root user with an explicit UID and adds permission to access the /usr/src/app folder
RUN adduser --disabled-password --gecos '' --uid 1001 nodeuser && chown -R nodeuser:nodeuser /usr/src/app
USER nodeuser

# Define the command to run your app
CMD [ "node", "dist/index.js" ]
