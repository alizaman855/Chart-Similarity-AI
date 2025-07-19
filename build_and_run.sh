#!/bin/bash

# Chart Analysis Docker Build and Run Script

echo "🐳 Chart Analysis Docker Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create data directories if they don't exist
echo -e "${YELLOW}📁 Creating data directories...${NC}"
mkdir -p data/{uploads,results,historical,temp}

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env file with your OpenAI API key before running.${NC}"
    
    # Prompt for OpenAI API key
    read -p "Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        sed -i "s/your_openai_api_key_here/$api_key/" .env
        echo -e "${GREEN}✅ OpenAI API key added to .env file${NC}"
    fi
fi

# Build the Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker-compose build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to build Docker image${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Ask user what to do
echo ""
echo "Choose an option:"
echo "1) Run the application"
echo "2) Run in development mode (with logs)"
echo "3) Just build (already done)"
echo "4) Stop and remove containers"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${YELLOW}🚀 Starting application...${NC}"
        docker-compose up -d
        echo -e "${GREEN}✅ Application started!${NC}"
        echo -e "${GREEN}🌐 Access the app at: http://localhost:5000${NC}"
        echo -e "${GREEN}🔧 Admin panel at: http://localhost:5000/admin${NC}"
        ;;
    2)
        echo -e "${YELLOW}🚀 Starting application in development mode...${NC}"
        docker-compose up
        ;;
    3)
        echo -e "${GREEN}✅ Build completed${NC}"
        ;;
    4)
        echo -e "${YELLOW}🛑 Stopping and removing containers...${NC}"
        docker-compose down
        echo -e "${GREEN}✅ Containers stopped and removed${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

# Show container status if running
if [ "$choice" == "1" ] || [ "$choice" == "2" ]; then
    echo ""
    echo -e "${YELLOW}📊 Container status:${NC}"
    docker-compose ps
    
    if [ "$choice" == "1" ]; then
        echo ""
        echo -e "${YELLOW}📋 Useful commands:${NC}"
        echo "  View logs:    docker-compose logs -f"
        echo "  Stop app:     docker-compose down"
        echo "  Restart:      docker-compose restart"
        echo "  Shell access: docker-compose exec chart-analysis bash"
    fi
fi