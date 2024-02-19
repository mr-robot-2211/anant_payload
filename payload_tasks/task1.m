% Read the image
img = imread('clocktower.jpeg');

% Display the original image
figure;
imshow(img);
title('Original Image');

% Convert the image to grayscale if it's a color image
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Perform pixel binning (average pooling)
binning_factor = 2; % You can adjust this factor as needed
binned_image = binning(img_gray, binning_factor);

% Display the binned image
figure;
imshow(uint8(binned_image));
title('Binned Image');

% Save the binned image if needed
imwrite(uint8(binned_image), 'binned_clocktower.jpg');

% Function for pixel binning (average pooling)
function binned_img = binning(img, factor)
    [rows, cols] = size(img);
    binned_rows = floor(rows / factor);
    binned_cols = floor(cols / factor);
    
    binned_img = zeros(binned_rows, binned_cols);
    
    for i = 1:binned_rows
        for j = 1:binned_cols
            row_range = (i - 1) * factor + 1 : i * factor;
            col_range = (j - 1) * factor + 1 : j * factor;
            
            block = img(row_range, col_range);
            binned_img(i, j) = mean(block(:));
        end
    end
end
