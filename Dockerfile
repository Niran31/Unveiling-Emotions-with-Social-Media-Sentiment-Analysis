# Use the official Python image
FROM python:3.10-slim

# Set up a new user to avoid running as root (Hugging Face Spaces often require this)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download the Spacy model explicitly during the build step
# This prevents downloading it during the first request
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY --chown=user . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Command to run the application using Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120", "--workers", "1", "--threads", "2"]
